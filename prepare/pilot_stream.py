"""N28 pilot selection and resumable streaming extraction, without full unpacking."""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import random
import re
import tarfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ("power_i", "power_s", "power_sca", "power_all", "power_t", "IR_drop")
ARCHIVES = {
    "IR_drop": [("IR_drop.tar.gz", "176QU64xifJC7XZpguFjMpj59Ug8d1MyL", 1261489584)],
    "power_all": [("power_all.tar.gz", "1-qI42_ayUgYQ_YXdBs8TXan9BKfGVTNh", 1775928904)],
    "power_i": [("power_i.tar.gz", "1Sh6FSCcfw3CKzylvw7p4CT-EUSgbVtWr", 1728315662)],
    "power_s": [("power_s.tar.gz", "1SVRMGYixRAD2uj74E3KPFZs3vXAEa1K4", 1733298112)],
    "power_sca": [("power_sca.tar.gz", "1tqWauzJH-UJyAGglgOw1_DKsaq5YLUZ4", 1776029346)],
    "power_t": [(f"power_t.tar.gz.{i:02d}", key, 2147483648 if i < 10 else 1461229014)
                for i, key in enumerate(("1G4Vw9oN70fpUBDT3jNcEJ58-5ACTJLHq", "1a8qRxdKjOjlYCr3mIxZPF5liL7GhpSC0",
                    "19qSuPw8x7TIek0FhDV8wPDnLtJZHq8yB", "1P7RPE78uLz-uez2F3mNNbLZeAY9TR-cr", "1IE-2_FWPRCcr11AQfFZ227z1sTHVW3Ll",
                    "1AAHDI6PAOU3DMfDMbQJ-BmA1LRDXlDKg", "1qUCW8p2dGOJI6H-1KXCtnDOZYo9IlxsS",
                    "1YMFY0AntNIchWO0qsawlEomEm0ehM91C", "1-nacdggSQ4t9tEH_YJVJo6F-dLAR_Hym", "1EjsNhlX01aPwR9APtnMJPNbjk8BRl1fZ",
                    "11UXvZedQaxDDX-bbvkSCiKlzUN8dEpos"))],
}
NAME = re.compile(r"^\d+-(RISCY(?:-FPU)?)-([ab])-(\d+)-c(\d+)-u(0\.\d+)-m(\d+|None)-p(\d+)-f([01])\.npy$")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as f:
        rows = [tuple(r) for r in csv.reader(f)]
    if any(len(r) != 2 for r in rows) or len({r[0] for r in rows}) != len(rows):
        raise ValueError("清单格式错误或重复样本")
    for row in rows:
        for name in row:
            p = PurePosixPath(name)
            if p.is_absolute() or ".." in p.parts:
                raise ValueError("清单路径越界")
    return rows


def describe(row):
    feature, label = map(PurePosixPath, row)
    if feature.name != label.name:
        raise ValueError("特征/标签样本名不一致")
    m = NAME.fullmatch(feature.name)
    if not m:
        raise ValueError(f"非目标N28训练族样本：{feature.name}")
    family, variant, macros, clock, util, placement, mesh, filler = m.groups()
    return {"name": feature.name, "family": family, "design": family + "-" + variant,
            "macros": macros, "clock": clock, "utilization": util, "placement": placement,
            "mesh": mesh, "filler": filler,
            "layout_group": "|".join((family, variant, macros, clock, util, placement)),
            "feature": str(feature), "label": str(label)}


def selection(source, test, seed=0):
    rows = read_rows(source)
    forbidden = {r[0] for r in read_rows(test)}
    if forbidden & {r[0] for r in rows}:
        raise ValueError("官方train/test样本重叠")
    pools = {}
    for row in rows:
        item = describe(row)
        pools.setdefault(item["design"], {}).setdefault(item["layout_group"], []).append(item)
    rng = random.Random(seed)
    selected = {"train": [], "validation": []}
    for design in ("RISCY-a", "RISCY-b", "RISCY-FPU-a", "RISCY-FPU-b"):
        groups = pools.get(design, {})
        if len(groups) < 40:
            raise ValueError(f"{design}不足40个不同布局组")
        available = sorted(groups)
        rng.shuffle(available)
        for split, count in (("validation", 8), ("train", 32)):
            covered = {key: set() for key in ("clock", "utilization", "macros", "placement")}
            for _ in range(count):
                def score(group):
                    item = groups[group][0]
                    return sum(item[k] not in covered[k] for k in covered)
                group = max(available, key=score)
                available.remove(group)
                item = dict(rng.choice(sorted(groups[group], key=lambda i: i["name"])))
                item["split"] = split
                selected[split].append(item)
                for key in covered:
                    covered[key].add(item[key])
    assert len(selected["train"]) == 128 and len(selected["validation"]) == 32
    assert not {x["layout_group"] for x in selected["train"]} & {x["layout_group"] for x in selected["validation"]}
    return {"seed": seed, "status": "selection_only_not_data_ready", "source_manifest_sha256": sha(source),
            "official_test_manifest_sha256": sha(test), "samples": selected,
            "selection_rule": "32 train + 8 validation per RTL variant; one sample per layout group; diversity-first deterministic sampling",
            "group_rule": "design variant/macros/clock/utilization/macro placement; power mesh and filler variants never cross splits",
            "scope": "development within seen RTL families, not unseen-design evaluation",
            "official_download_folder": "https://drive.google.com/drive/folders/18DzI3RRYDF0T5VlMwmkYdFx0kVuW3sql"}


class CachedFile:
    def __init__(self, metadata, cache, progress):
        self.name, self.file_id, self.size = metadata
        self.path = cache / (self.name + ".partial")
        self.handle = self.path.open("a+b")
        self.end = self.handle.seek(0, 2)
        if self.end > self.size:
            raise ValueError("下载缓存大于官方文件大小")
        self.pos = 0
        self.response = None
        self.progress = progress

    def read(self, n):
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        if self.pos < self.end:
            self.handle.seek(self.pos)
            data = self.handle.read(min(n, self.end - self.pos))
        else:
            import shutil
            if shutil.disk_usage(self.path.parent).free < 3 * 1024**3:
                raise OSError("磁盘余量不足3GiB，暂停流式下载，保留缓存")
            for attempt in range(5):
                try:
                    if self.response is None:
                        url = f"https://drive.usercontent.google.com/download?id={self.file_id}&export=download&confirm=t"
                        req = urllib.request.Request(url, headers={"Range": f"bytes={self.pos}-", "User-Agent": "Mozilla/5.0"})
                        self.response = urllib.request.urlopen(req, timeout=90)
                        content = self.response.headers.get("Content-Type", "")
                        span = self.response.headers.get("Content-Range", "")
                        if "text/html" in content or (span and not span.startswith(f"bytes {self.pos}-")) or (self.pos and not span):
                            raise ValueError("Google Drive未返回正确二进制范围；不能把网页保存成数据")
                        if span and not span.endswith(f"/{self.size}"):
                            raise ValueError("官方文件大小发生变化，不能混用缓存")
                    data = self.response.read(n)
                    if not data:
                        raise IOError("远端文件提前结束")
                    break
                except (OSError, TimeoutError):
                    if self.response:
                        self.response.close()
                    self.response = None
                    if attempt == 4:
                        raise
                    time.sleep(min(2 ** attempt, 15))
            self.handle.seek(0, 2)
            self.handle.write(data)
            self.handle.flush()
            self.end += len(data)
        self.pos += len(data)
        self.progress(self.name, self.pos, self.size)
        return data

    def close(self):
        if self.response:
            self.response.close()
        self.handle.close()


class PartsStream:
    def __init__(self, files):
        self.files = files
        self.index = 0
    def read(self, size):
        data = bytearray()
        while len(data) < size and self.index < len(self.files):
            chunk = self.files[self.index].read(size - len(data))
            if not chunk:
                self.index += 1
            else:
                data.extend(chunk)
        return bytes(data)


def extract_component(component, names, raw, cache, progress):
    target = raw / component
    target.mkdir(parents=True, exist_ok=True)
    receipts_path = target / "RECEIPTS.json"
    receipts = json.loads(receipts_path.read_text(encoding="utf-8")) if receipts_path.exists() else {}
    present = {name for name in names if (target / name).is_file() and receipts.get(name, {}).get("sha256") == sha(target / name)}
    needed = set(names) - present
    if not needed:
        return receipts
    files = [CachedFile(item, cache, progress) for item in ARCHIVES[component]]
    try:
        decoded = io.BufferedReader(gzip.GzipFile(fileobj=PartsStream(files), mode="rb"), buffer_size=1024 * 1024)
        layers = 1
        while decoded.peek(2)[:2] == b"\x1f\x8b":
            if layers >= 3:
                raise ValueError("归档嵌套gzip超过三层")
            decoded = io.BufferedReader(gzip.GzipFile(fileobj=decoded, mode="rb"), buffer_size=1024 * 1024)
            layers += 1
        save(target / "ARCHIVE_FORMAT.json", {"gzip_layers": layers})
        with tarfile.open(fileobj=decoded, mode="r|", bufsize=1024 * 1024) as archive:
            for member in archive:
                if not member.isfile():
                    continue
                path = PurePosixPath(member.name)
                sample_name = path.name if path.name.endswith(".npy") else path.name + ".npy"
                if sample_name not in needed:
                    continue
                if len(path.parts) < 2 or path.parts[-2] != component or member.size > 512 * 1024**2:
                    raise ValueError("归档成员路径或大小不符合约定")
                dest = target / sample_name
                temporary = dest.with_suffix(".npy.partial")
                with archive.extractfile(member) as source, temporary.open("wb") as output:
                    shutil_copy(source, output)
                if temporary.stat().st_size != member.size:
                    raise IOError("抽取成员长度不完整")
                temporary.replace(dest)
                receipts[sample_name] = {"bytes": member.size, "sha256": sha(dest), "archive_member": member.name}
                save(receipts_path, receipts)
                needed.remove(sample_name)
                progress(component, len(names) - len(needed), len(names), extracted=True)
                if not needed:
                    break
        if needed:
            raise ValueError(f"{component}缺少{len(needed)}个目标样本")
    finally:
        for file in files:
            file.close()
    return receipts


def shutil_copy(source, output):
    while True:
        block = source.read(1024 * 1024)
        if not block:
            return
        output.write(block)


def preprocess(plan, raw, data):
    import numpy as np
    import cv2
    def resized(array):
        return cv2.resize(array, (256, 256), interpolation=cv2.INTER_AREA)
    def normalized(array):
        if array.max() == 0:
            return array
        if array.max() == array.min():
            raise ValueError("非零常量特征会触发官方min-max除零，必须明确处理，不能静默改配方")
        return (array - array.min()) / (array.max() - array.min())
    (data / "feature").mkdir(parents=True, exist_ok=True)
    (data / "label").mkdir(parents=True, exist_ok=True)
    for item in plan["samples"]["train"] + plan["samples"]["validation"]:
        name = item["name"]
        inputs = []
        for component in COMPONENTS[:5]:
            array = np.load(raw / component / name, allow_pickle=False)
            if not np.isfinite(array).all():
                raise ValueError("原始特征非有限值")
            if component == "power_t":
                if array.ndim != 3 or array.shape[0] != 20:
                    raise ValueError("power_t须为20个时间切片")
                inputs.extend(normalized(resized(part)) for part in array)
            else:
                inputs.append(normalized(resized(array.squeeze())))
        label = np.load(raw / "IR_drop" / name, allow_pickle=False).squeeze()
        if not np.isfinite(label).all():
            raise ValueError("原始标签非有限值")
        label = (np.log10(resized(np.clip(label, 1e-6, 50))) + 6) / (np.log10(50) + 6)
        feature = np.array(inputs).transpose(1, 2, 0)
        label = label[:, :, None]
        for kind, array in (("feature", feature), ("label", label)):
            path = data / kind / name
            if path.exists():
                existing = np.load(path, allow_pickle=False)
                if not np.array_equal(existing, array):
                    raise ValueError("已有处理结果不一致，不覆盖")
            else:
                with path.with_suffix(".npy.partial").open("wb") as f:
                    np.save(f, array, allow_pickle=False)
                path.with_suffix(".npy.partial").replace(path)


def audit(plan, data):
    import numpy as np
    missing, files = [], {}
    for item in plan["samples"]["train"] + plan["samples"]["validation"]:
        for kind, expected in (("feature", (256, 256, 24)), ("label", (256, 256, 1))):
            path = data / kind / item["name"]
            if not path.is_file():
                missing.append(str(path))
                continue
            a = np.load(path, allow_pickle=False)
            if a.shape != expected or not np.isfinite(a).all() or a.min() < -1e-6 or a.max() > 1 + 1e-6:
                raise ValueError(f"数组shape/范围/有限值核验失败：{path}")
            files[f"{kind}/{item['name']}"] = {"bytes": path.stat().st_size, "sha256": sha(path), "dtype": str(a.dtype), "min": float(a.min()), "max": float(a.max())}
    return {"ready": not missing, "train_samples": 128, "validation_samples": 32, "missing": missing,
            "files": files, "processed_label_unit": "official logarithmic normalized scale; raw physical unit requires source confirmation"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--official-test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    planned = selection(args.source, args.official_test, args.seed)
    plan_path = args.output / "SELECTION.json"
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != planned:
        raise ValueError("已有抽样协议不同，必须使用新输出目录")
    save(plan_path, planned)
    for split in ("train", "validation"):
        with (args.output / f"{split}.csv").open("w", encoding="utf-8", newline="") as f:
            csv.writer(f, lineterminator="\n").writerows((i["feature"], i["label"]) for i in sorted(planned["samples"][split], key=lambda i: i["name"]))
    (args.output / "sample_names.txt").write_text("\n".join(sorted(i["name"] for items in planned["samples"].values() for i in items)) + "\n", encoding="utf-8")
    save(args.output / "DOWNLOADS.json", ARCHIVES)
    print("128/32分组清单已生成：", plan_path, flush=True)
    if args.plan_only:
        return
    import shutil
    if shutil.disk_usage(args.output).free < 8 * 1024**3:
        raise ValueError("可用空间不足8GiB，停止，不删除任何原文件")
    cache, raw = args.output / "archive_cache", args.output / "raw_selected"
    cache.mkdir(exist_ok=True)
    raw.mkdir(exist_ok=True)
    last = [0.]
    def progress(name, done, total, extracted=False):
        record = {"phase": "extracting", "component_or_archive": name, "done": done, "total": total, "extracted": extracted, "time": time.time()}
        if time.monotonic() - last[0] > 15 or extracted:
            save(args.output / "PROGRESS.json", record)
            print(name, f"{done}/{total}" if extracted else f"{done / 1024**2:.1f}/{total / 1024**2:.1f} MiB", flush=True)
            last[0] = time.monotonic()
    names = {i["name"] for items in planned["samples"].values() for i in items}
    from sa0.session import session_lock
    with session_lock(args.output):
        try:
            for component in COMPONENTS:
                extract_component(component, names, raw, cache, progress)
            save(args.output / "PROGRESS.json", {"phase": "preprocessing"})
            preprocess(planned, raw, args.output / "data")
            result = audit(planned, args.output / "data")
            result["selection_sha256"] = sha(plan_path)
            result["preprocessor_sha256"] = sha(Path(__file__))
            save(args.output / "DATASET.json", result)
            save(args.output / "PROGRESS.json", {"phase": "completed", "ready": result["ready"]})
            print("数据已完成并通过核验：", args.output / "DATASET.json", flush=True)
        except BaseException as error:
            save(args.output / "FAILURE.json", {"error": f"{type(error).__name__}: {error}", "time": time.time(), "partial_files_preserved": True})
            raise


if __name__ == "__main__":
    main()
