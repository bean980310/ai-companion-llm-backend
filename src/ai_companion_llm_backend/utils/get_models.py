import os

from ..logging import logger

def has_required_files(filenames, required_names=None, required_extensions=None):
    """
    파일 목록(filenames) 중에 필수 파일명이 존재하거나,
    필수 확장자(extension)를 가진 파일이 존재하는지 확인.
    기본적으로 config.json이나 (.safetensors, .ckpt, .pt, .pth) 파일을 체크.
    """
    if required_names is None:
        required_names = {"config.json"}
    if required_extensions is None:
        required_extensions = {".safetensors", ".ckpt", ".pt", ".pth"}
    for fname in filenames:
        lower_fname = fname.lower()
        if lower_fname in required_names:
            return True
        for ext in required_extensions:
            if lower_fname.endswith(ext):
                return True
    return False

def scan_files_with_extension(root: str, allowed_extensions: set) -> list:
    """
    지정된 root 디렉토리부터 재귀적으로 파일을 탐색하여,
    allowed_extensions에 해당하는 파일들의 상대경로 목록을 반환.
    """
    results = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if any(fname.lower().endswith(ext) for ext in allowed_extensions) and not fname.startswith("."):
                # root로부터의 상대경로 (하위 디렉토리 포함)
                rel_path = os.path.relpath(os.path.join(dirpath, fname), root)
                results.append(rel_path)
    return results

def scan_local_models(root="./models/llm", model_type=None):
    """
    로컬 llm 모델들을 스캔.
    - model_type이 지정되지 않으면, 'transformers', 'gguf', 'mlx' 폴더를 각각 스캔.
    - gguf의 경우 gguf 파일 자체를, 그 외에는 디렉토리 단위(필요한 파일이 있는 경우)로 등록.
    """
    if not os.path.isdir(root):
        os.makedirs(root, exist_ok=True)
    
    local_models = []
    # model_type이 지정되지 않았다면, 세부 폴더 목록: transformers, gguf, mlx
    subdirs = ['transformers', 'gguf', 'mlx'] if model_type is None else [model_type]
    
    for subdir in subdirs:
        subdir_path = os.path.join(root, subdir)
        if not os.path.isdir(subdir_path):
            continue
        
        if subdir == "gguf":
            # gguf는 파일 단위로 스캔 (확장자가 .gguf 인 파일)
            files = scan_files_with_extension(subdir_path, {".gguf"})
            for f in files:
                model_id = os.path.join(subdir, f)
                local_models.append({"model_id": model_id, "model_type": subdir})
        else:
            # transformers와 mlx는 기존대로, 해당 폴더 내에 필요한 파일이 있으면 디렉토리 단위로 등록
            for dirpath, _, filenames in os.walk(subdir_path):
                if has_required_files(filenames, required_extensions={".safetensors", ".ckpt", ".pt", ".pth"}):
                    rel_path = os.path.relpath(dirpath, subdir_path)
                    model_id = subdir if rel_path == "." else os.path.join(subdir, rel_path)
                    local_models.append({"model_id": model_id, "model_type": subdir})
    
    logger.info(f"Scanned local models: {local_models}")
    return local_models

def get_all_local_models():
    """모든 모델 유형별 로컬 모델 목록을 가져옴"""
    models = scan_local_models()  # 모든 유형 스캔
    transformers = [m["model_id"] for m in models if m["model_type"] == "transformers"]
    gguf = [m["model_id"] for m in models if m["model_type"] == "gguf"]
    mlx = [m["model_id"] for m in models if m["model_type"] == "mlx"]
    return {
        "transformers": transformers,
        "gguf": gguf,
        "mlx": mlx
    }