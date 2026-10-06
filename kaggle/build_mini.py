"""Build a private Kaggle input archive and an offline B0 notebook (stdlib only)."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(
        ['git', '-c', f'safe.directory={ROOT.as_posix()}', '-C', str(ROOT), *args]
    ).decode().strip()


def cell(kind, source):
    result = {'cell_type': kind, 'metadata': {},
              'source': textwrap.dedent(source).strip().splitlines(keepends=True)}
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
        compile(''.join(result['source']), '<notebook>', 'exec')
    return result


def build(data_root, output):
    output.mkdir(parents=True, exist_ok=False)
    commit = git('rev-parse', 'HEAD')
    if git('status', '--porcelain', '--untracked-files=no'):
        raise RuntimeError('Commit tracked changes before packaging a frozen source version')
    files = {}
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / 'source.bundle'
        git('bundle', 'create', str(bundle), 'HEAD')
        archive = output / 'IR_AutoResearch_kaggle_mini.zip'
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            z.write(bundle, 'source.bundle')
            files['source.bundle'] = sha(bundle)
            for manifest in ['smoke_train.csv', 'smoke_validation.csv']:
                with (ROOT / 'prepare' / 'manifests' / manifest).open(newline='') as f:
                    for row in csv.reader(f):
                        for relative in row:
                            path = (data_root / relative).resolve()
                            if not path.is_relative_to(data_root.resolve()):
                                raise ValueError('Data path escapes root')
                            name = 'data/' + relative
                            if name not in files:
                                z.write(path, name)
                                files[name] = sha(path)
            z.writestr('PACKAGE.json', json.dumps(
                {'source_commit': commit, 'scope': 'mini', 'train_samples': 3,
                 'validation_samples': 1, 'files_sha256': files}, indent=2))
    cells = [cell('markdown', '''
        # B0 mini：Kaggle 单 GPU 验证
        将配套 zip 上传为 **Private Dataset**，添加到此 **Private Notebook**。
        在 Settings 中启用 GPU，再按顺序运行。无需 GitHub token 或 Internet。
        配方保持 B0：100 步、batch=2、seed=0，学习率周期仍为 200000 步。
        这是 3 train / 1 validation 的工程检查，不能用于论文成绩或泛化结论。
        checkpoint 能力取决于冻结源码版本；先查 B0 README，暂勿启动长训练。
    '''), cell('code', '''
        from pathlib import Path
        import os, sys, json, hashlib, zipfile, subprocess
        from datetime import datetime, timezone
        INPUT = Path('/kaggle/input')
        WORK = Path('/kaggle/working')
        STEPS = 100
        run_name = 'b0_mini_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        RUN = WORK / run_name
        RUN.mkdir(parents=True, exist_ok=False)
        def sha(path):
            h = hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda: f.read(1024 * 1024), b''):
                    h.update(block)
            return h.hexdigest()
    '''), cell('code', '''
        import torch, numpy, cv2
        print('Python:', sys.version)
        print('torch / numpy / cv2:', torch.__version__, numpy.__version__, cv2.__version__)
        if not torch.cuda.is_available():
            raise RuntimeError('GPU 未启用：在 Notebook Settings 中启用 GPU 后重试')
        for i in range(torch.cuda.device_count()):
            print('GPU', i, torch.cuda.get_device_name(i),
                  torch.cuda.get_device_properties(i).total_memory / 1024**3, 'GiB')
        print('使用 cuda:0；两块 GPU 的显存不会自动合并。')
    '''), cell('code', '''
        archives = list(INPUT.rglob('IR_AutoResearch_kaggle_mini.zip'))
        if len(archives) == 1:
            payload = RUN / 'input'
            payload.mkdir()
            with zipfile.ZipFile(archives[0]) as z:
                for member in z.infolist():
                    target = (payload / member.filename).resolve()
                    if not target.is_relative_to(payload.resolve()):
                        raise ValueError('Invalid archive path')
                z.extractall(payload)
        elif len(archives) == 0:
            # Kaggle can expand an uploaded archive into Dataset files.
            packages = list(INPUT.rglob('PACKAGE.json'))
            if len(packages) != 1:
                raise RuntimeError('请添加且只添加一份配套 Dataset')
            payload = packages[0].parent
        else:
            raise RuntimeError('找到多份输入 zip，请只添加一份配套 Dataset')
        package = json.loads((payload / 'PACKAGE.json').read_text())
        for relative, expected in package['files_sha256'].items():
            if not (payload / relative).resolve().is_relative_to(payload.resolve()):
                raise ValueError('Invalid package path')
            if sha(payload / relative) != expected:
                raise RuntimeError('文件校验失败：' + relative)
        repo = RUN / 'repo'
        subprocess.run(['git', 'clone', str(payload / 'source.bundle'), str(repo)], check=True)
        head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip()
        assert head == package['source_commit'], (head, package['source_commit'])
        print('冻结代码版本：', head, '数据：3 train / 1 validation')
    '''), cell('code', '''
        output = RUN / 'result'
        command = [sys.executable, '-m', 'b0.run',
                   '--data-root', str(payload / 'data'),
                   '--train-manifest', str(repo / 'prepare/manifests/smoke_train.csv'),
                   '--validation-manifest', str(repo / 'prepare/manifests/smoke_validation.csv'),
                   '--output', str(output), '--scope', 'mini', '--steps', str(STEPS),
                   '--lr-horizon-steps', '200000', '--batch-size', '2', '--seed', '0',
                   '--device', 'cuda', '--num-workers', '0', '--cpu-threads', '2',
                   '--log-every', '10', '--check-values']
        environment = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                           CUDA_VISIBLE_DEVICES='0')
        subprocess.run(command, cwd=repo, env=environment, check=True)
    '''), cell('code', '''
        result = json.loads((output / 'result.json').read_text())
        assert result['status'] == 'completed' and result['checkpoint_reload_passed']
        print(json.dumps({k: result[k] for k in
                          ['steps', 'initial_validation', 'validation', 'resources']}, indent=2))
        (output / 'kaggle_package.json').write_text(json.dumps(package, indent=2))
        destination = WORK / (run_name + '_outputs.zip')
        with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as z:
            for path in sorted(output.rglob('*')):
                if path.is_file():
                    z.write(path, path.relative_to(output))
        print('下载结果：', destination, 'SHA256:', sha(destination))
    '''), cell('markdown', '''
        完成后用 Save Version 保存运行与输出，再下载 `_outputs.zip`。
        保留 result.json、environment.json（若存在）、training.jsonl、源码快照和权重。
        将 result.json 返回本地，我们先看速度、峰值显存和 reload 验证，再决定训练预算。
        中断时请保存 failure.json；续训能力以冻结版本的 B0 README 为准。
    ''')]
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3',
                'language': 'python', 'name': 'python3'}}, 'nbformat': 4, 'nbformat_minor': 4}
    (output / 'B0_mini.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding='utf-8')
    report = {'archive': archive.name, 'sha256': sha(archive), 'bytes': archive.stat().st_size,
              'source_commit': commit, 'data_files': len(files) - 1}
    (output / 'BUILD.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.data_root, args.output)
