from abc import ABC, abstractmethod
from typing import Any, Optional

import transformers.utils.logging as logging

import infinicore

logger = logging.get_logger(__name__)


class CacheLayerMixin(ABC):
    """Base, abstract class for a single layer's cache."""

    def __init__(self):
        self.keys, self.values = None, None

    def __repr__(self):
        return f"{self.__class__.__name__}"

    @abstractmethod
    def lazy_initialization(self, key_states: infinicore.Tensor): ...

    @abstractmethod
    def update(
        self,
        key_states: infinicore.Tensor,
        value_states: infinicore.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[infinicore.Tensor, infinicore.Tensor]: ...


class DynamicLayer(CacheLayerMixin):
    """
    A cache layer that grows dynamically as more tokens are generated. This is the default for generative models.
    It stores the key and value states as tensors of shape `[batch_size, seq_len, num_heads,  head_dim]`.
    """

    def lazy_initialization(self, key_states: infinicore.Tensor):
        batch_size, seq_len, num_heads, head_dim = key_states.shape

        if self.keys is None:
            dtype, device = key_states.dtype, key_states.device

            self.max_seq_len = max(8, seq_len)
            self.cache_position = 0

            self.keys = infinicore.empty(
                [batch_size, self.max_seq_len, num_heads, head_dim],
                dtype=dtype,
                device=device,
            )
            self.values = infinicore.empty(
                [batch_size, self.max_seq_len, num_heads, head_dim],
                dtype=dtype,
                device=device,
            )
        elif self.cache_position + seq_len >= self.max_seq_len:
            dtype, device = key_states.dtype, key_states.device

            self.max_seq_len = max(self.max_seq_len * 2, self.cache_position + seq_len)

            keys_new = infinicore.empty(
                [batch_size, self.max_seq_len, num_heads, head_dim],
                dtype=dtype,
                device=device,
            )
            values_new = infinicore.empty(
                [batch_size, self.max_seq_len, num_heads, head_dim],
                dtype=dtype,
                device=device,
            )
            keys_new.narrow(1, 0, self.cache_position).copy_(
                self.keys.narrow(1, 0, self.cache_position)
            )
            values_new.narrow(1, 0, self.cache_position).copy_(
                self.values.narrow(1, 0, self.cache_position)
            )

            self.keys, self.values = keys_new, values_new

    def update(
        self,
        key_states: infinicore.Tensor,
        value_states: infinicore.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ):
        # Lazy initialization
        self.lazy_initialization(key_states)

        _, seq_len, _, _ = key_states.shape
        index = self.cache_position

        # Update the cache
        self.keys.narrow(1, index, seq_len).copy_(key_states)
        self.values.narrow(1, index, seq_len).copy_(value_states)
        self.cache_position += seq_len

        return self.keys.narrow(1, 0, self.cache_position), self.values.narrow(
            1, 0, self.cache_position
        )


class Cache:
    """
    A `Cache` is mostly a list of `CacheLayerMixin` objects, one per model layer. It serves as a container for
    the Cache of each layer.

    Args:
        layers (`Optional`, *optional*):
            A list of pre-created `CacheLayerMixin`. If omitted (`None`), then `layer_class_to_replicate` will
            be used.
        layer_class_to_replicate (`type[CacheLayerMixin]`, *optional*):
            Only used if `layers` is omitted (`None`), in which case it will be used as the base class for each layer,
            and the layers will be added lazily as soon as `update` is called with a `layer_idx` greater than the current
            list of layers.
    """

    def __init__(
        self,
        layers: Optional[list[CacheLayerMixin]] = None,
        layer_class_to_replicate: Optional[type[CacheLayerMixin]] = None,
    ):
        if layers is not None and layer_class_to_replicate is not None:
            raise ValueError(
                "You can construct a Cache either from a list `layers` of all the predefined `CacheLayer`, or from a "
                "`layer_class_to_replicate`, in which case the Cache will append a new layer corresponding to "
                "`layer_class_to_replicate` for each new call to `update` with an idx not already in the Cache."
            )
        if layers is None and layer_class_to_replicate is None:
            raise ValueError(
                "You should provide exactly one of `layers` or `layer_class_to_replicate` to initialize a Cache."
            )

        self.layers = layers if layers is not None else []
        self.layer_class_to_replicate = layer_class_to_replicate

    def update(
        self,
        key_states: infinicore.Tensor,
        value_states: infinicore.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[infinicore.Tensor, infinicore.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`infinicore.Tensor`):
                The new key states to cache.
            value_states (`infinicore.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`dict[str, Any]`, *optional*):
                Additional arguments for the cache subclass. These are specific to each subclass and allow new types of
                cache to be created.

        Return:
            A tuple containing the updated key and value states.
        """

        # In this case, the `layers` were not provided, and we must append as much as `layer_idx`
        if self.layer_class_to_replicate is not None:
            while len(self.layers) <= layer_idx:
                self.layers.append(self.layer_class_to_replicate())

        keys, values = self.layers[layer_idx].update(
            key_states, value_states, cache_kwargs
        )

        return keys.contiguous(), values.contiguous()


class DynamicCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.
    It stores the key and value states as a list of `CacheLayer`, one for each layer. The expected shape for each tensor
    in the `CacheLayer`s is `[batch_size, num_heads, seq_len, head_dim]`.
    If a config is passed, it will additionally check for sliding or hybrid cache structure, greatly reducing the
    memory requirement of the cached tensors to `[batch_size, num_heads, min(seq_len, sliding_window), head_dim]`.

    See `Cache` for details on common methods that are implemented by all cache classes.

    Args:
        config (`PretrainedConfig`, *optional*):
            The config of the model for which this Cache will be used. If passed, it will be used to check for sliding
            or hybrid layer structure, greatly reducing the memory requirement of the cached tensors to
            `[batch_size, num_heads, min(seq_len, sliding_window), head_dim]`.

    Example:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

    >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
    >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

    >>> inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

    >>> # Prepare a cache class and pass it to model's forward
    >>> past_key_values = DynamicCache(config=model.config)
    >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
    >>> outputs.past_key_values # access cache filled with key/values from generation
    ```
    """

    def __init__(
        self,
        config: Optional["Config"] = None,
    ):
        layers = []
        # If a config is passed, use it to infer the layer types and initialize accordingly
        if config is not None:
            config = config.get_text_config()

            layer_types = None
            if layer_types is None:
                layer_types = [
                    "full_attention" for _ in range(config.num_hidden_layers)
                ]

            for layer_type in layer_types:
                layers.append(DynamicLayer())

        # If neither of config nor ddp_data was passed, then simply lazy init a full cache of DynamicLayer
        if len(layers) == 0:
            super().__init__(
                layer_class_to_replicate=DynamicLayer,
            )
        else:
            super().__init__(
                layers=layers,
            )
