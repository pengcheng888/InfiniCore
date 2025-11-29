import time

import infinicore
from infinicore.lib import _infinicore

_TORCH_DEVICE_MAP = {
    _infinicore.Device.Type.CPU: "cpu",
    _infinicore.Device.Type.NVIDIA: "cuda",
    _infinicore.Device.Type.CAMBRICON: "mlu",
    _infinicore.Device.Type.ASCEND: "npu",
    _infinicore.Device.Type.METAX: "cuda",
    _infinicore.Device.Type.MOORE: "musa",
    _infinicore.Device.Type.ILUVATAR: "cuda",
    _infinicore.Device.Type.KUNLUN: "cuda",
    _infinicore.Device.Type.HYGON: "cuda",
    _infinicore.Device.Type.QY: "cuda",
}

# -------------------------------------------------------------------------------- #
#
# -------------------------------------------------------------------------------- #

all_device_types = tuple(_infinicore.Device.Type.__members__.values())[:-1]
all_device_count = tuple(
    _infinicore.get_device_count(device) for device in all_device_types
)

print("?? all_device_types:", all_device_types)
print("?? all_device_count:", all_device_count)


# -------------------------------------------------------------------------------- #
#         infinicore_2_python, python_2_infinicore
# -------------------------------------------------------------------------------- #
infinicore_2_python_dict = {}
python_2_infinicore_dict = {}
for infinicore_devices_type, count in zip(all_device_types, all_device_count):
    if 0 == count:
        continue
    print("(+++++++++++++++++++++++++++++++++++++)")
    print("Device Type:", infinicore_devices_type)
    print("Device Count:", count)
    print("")
    for i in range(count):
        #
        infinicore_devices_type = infinicore_devices_type
        infinicore_devices_index = i
        python_devices_type = _TORCH_DEVICE_MAP[infinicore_devices_type]
        python_devices_index = i

        if infinicore_2_python_dict.get(infinicore_devices_type) is not None:
            infinicore_2_python_dict[infinicore_devices_type].append(
                (python_devices_type, python_devices_index)
            )

        else:
            infinicore_2_python_dict[infinicore_devices_type] = [
                (python_devices_type, python_devices_index)
            ]

        if python_2_infinicore_dict.get(python_devices_type) is not None:
            python_2_infinicore_dict[python_devices_type].append(
                (infinicore_devices_type, infinicore_devices_index)
            )
        else:
            python_2_infinicore_dict[python_devices_type] = [
                (infinicore_devices_type, infinicore_devices_index)
            ]


print("-----------------++++++++++++++++++++++++")
print(infinicore_2_python_dict)
print(python_2_infinicore_dict)


class device:
    s_time = 0
    s_count = 0

    _underlying: _infinicore.Device
    type: str
    index: int

    def __init__(self, type=None, index=None):
        device.s_count += 1
        t1 = time.time()

        if isinstance(type, device):
            self.type = type.type
            self.index = type.index
            return

        if type is None:
            type = "cpu"

        if ":" in type:
            if index is not None:
                raise ValueError(
                    '`index` should not be provided when `type` contains `":"`.'
                )

            type, index = type.split(":")
            index = int(index)

        self.type = type
        self.index = index

        # _type, _index = device._to_infinicore_device(type, index if index else 0)
        # self._underlying = _infinicore.Device(_type, _index)

        # self._underlying = self.to_infinicore_device()

        t2 = time.time()
        device.s_time += t2 - t1

    @property
    def _underlying(self):
        return self.to_infinicore_device()

    def __repr__(self):
        return f"device(type='{self.type}'{f', index={self.index}' if self.index is not None else ''})"

    def __str__(self):
        return f"{self.type}{f':{self.index}' if self.index is not None else ''}"

    def __eq__(self, other):
        """
        Compare two device objects for equality.

        Args:
            other: The object to compare with

        Returns:
            bool: True if both objects are device instances with the same type and index
        """
        if not isinstance(other, device):
            return False
        return self.type == other.type and self.index == other.index

    @staticmethod
    def _to_infinicore_device(type, index):
        all_device_types = tuple(_infinicore.Device.Type.__members__.values())[:-1]
        all_device_count = tuple(
            _infinicore.get_device_count(device) for device in all_device_types
        )

        torch_devices = {
            torch_type: {
                infinicore_type: 0
                for infinicore_type in all_device_types
                if _TORCH_DEVICE_MAP[infinicore_type] == torch_type
            }
            for torch_type in _TORCH_DEVICE_MAP.values()
        }

        for i, count in enumerate(all_device_count):
            infinicore_device_type = _infinicore.Device.Type(i)
            torch_devices[_TORCH_DEVICE_MAP[infinicore_device_type]][
                infinicore_device_type
            ] += count

        for infinicore_device_type, infinicore_device_count in torch_devices[
            type
        ].items():
            for i in range(infinicore_device_count):
                if index == 0:
                    return infinicore_device_type, i

                index -= 1

    # @staticmethod
    # def _from_infinicore_device_old(infinicore_device):
    #     type = _TORCH_DEVICE_MAP[infinicore_device.type]

    #     base_index = 0

    #     for infinicore_type, torch_type in _TORCH_DEVICE_MAP.items():
    #         if torch_type != type:
    #             continue

    #         if infinicore_type == infinicore_device.type:
    #             break

    #         base_index += _infinicore.get_device_count(infinicore_type)

    #     return device(type, base_index + infinicore_device.index)

    @staticmethod
    def from_infinicore_device(infinicore_device: _infinicore.Device):
        type, index = infinicore_2_python_dict[infinicore_device.type][
            infinicore_device.index
        ]
        return device(type, index)

    def to_infinicore_device(self):
        type, index = python_2_infinicore_dict[self.type][self.index]
        return _infinicore.Device(type, index)
