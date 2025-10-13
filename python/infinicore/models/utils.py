import infinicore
from collections import OrderedDict


class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


ACT2CLS = {

    "silu": infinicore.nn.SiLU,

}
ACT2FN = ClassInstantier(ACT2CLS)
