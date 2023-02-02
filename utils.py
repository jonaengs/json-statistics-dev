from typing import Callable


def deep_dict_update(target: dict, data: dict, modification_only=True) -> None:
    """Recursively update the target dict using the data dict"""
    for k in data:
        if k not in target and modification_only:
            raise ValueError(f"data key {k} not present in target dict. data dict must be a subset of target dict")
        elif isinstance(target[k], dict) and isinstance(data[k], dict):  # Use isinstance to be compatible with Munch objects (which subclass dict)
            deep_dict_update(target[k], data[k])
        else:
            target[k] = data[k]


def get_time_formatter(t) -> Callable[[int|float], str]:
    """ takes a time in ns and finds the best format (s, ms, μs, ns)"""
    i = 0
    while t >= 1000 and i < 3:
        t /= 1000
        i += 1

    sizes = ["ns", "us", "ms", "s"]
    return lambda _t : f"{_t//(1000**i):.0f}{sizes[i]}"


if __name__ == '__main__':
    target_dict = {
        1: {
            1: "abc",
            2: []
        },
        2: [1, 2, 3],
        3: 3,
        4: {
            1: {1: "abc"},
            2: {1: "abc"},
        }
    }

    update_dict = {
        1: {
            2: "abc"
        },
        2: [1, 2],
        4: {
            2: {1: "updated!"}
        }
    }

    result_dict = {
        1: {
            1: "abc",
            2: "abc"
        },
        2: [1, 2],
        3: 3,
        4: {
            1: {1: "abc"},
            2: {1: "updated!"},
        }
    }

    assert target_dict != result_dict
    deep_dict_update(target_dict, update_dict)
    assert target_dict == result_dict


    print("utils tests passed!")
