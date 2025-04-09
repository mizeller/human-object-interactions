from loguru import logger


def log(d: dict, indent: int = 0) -> None:
    def get_max_lens(d: dict, prefix: str = "") -> tuple:
        key_lens, shape_lens, type_lens, device_lens = [], [], [], []
        for k, v in d.items():
            full_key = f"{prefix}{k}"
            if isinstance(v, dict):
                sub_lens = get_max_lens(v, f"{full_key}.")
                key_lens.extend(sub_lens[0])
                shape_lens.extend(sub_lens[1])
                type_lens.extend(sub_lens[2])
                device_lens.extend(sub_lens[3])
            else:
                key_lens.append(len(full_key))
                try:
                    shape_lens.append(len(str(v.shape)))
                    type_lens.append(len(str(type(v))))
                    device_lens.append(len(str(v.device)))
                except AttributeError:
                    shape_lens.append(0)
                    type_lens.append(len(str(type(v))))
                    device_lens.append(0)
        return key_lens, shape_lens, type_lens, device_lens

    def log_recursive(d: dict, prefix: str = "", indent: int = 0) -> None:
        for k, v in d.items():
            full_key = f"{prefix}{k}"
            if isinstance(v, dict):
                logger.info(" " * indent + f"{full_key}:")
                log_recursive(v, f"{full_key}.", indent + 2)
            else:
                try:
                    logger.info(
                        " " * indent
                        + fmt.format(
                            full_key,
                            str(v.shape),
                            str(type(v)),
                            str(v.dtype),
                            str(v.device),
                            key_len=max_key_len,
                            shape_len=max_shape_len,
                            type_len=max_type_len,
                            device_len=max_device_len,
                        )
                    )
                except AttributeError:
                    # For non-tensor objects
                    logger.info(
                        " " * indent
                        + fmt.format(
                            full_key,
                            "N/A",
                            str(type(v)),
                            "N/A",
                            "N/A",
                            key_len=max_key_len,
                            shape_len=max_shape_len,
                            type_len=max_type_len,
                            device_len=max_device_len,
                        )
                    )

    # Get maximum lengths for all nested items
    key_lens, shape_lens, type_lens, device_lens = get_max_lens(d)
    max_key_len = max(key_lens) if key_lens else 0
    max_shape_len = max(shape_lens) if shape_lens else 0
    max_type_len = max(type_lens) if type_lens else 0
    max_device_len = max(device_lens) if device_lens else 0

    fmt = "{:<{key_len}}  {:<{shape_len}}  {:<{type_len}}  {:<{device_len}}  {}"
    log_recursive(d, indent=indent)


