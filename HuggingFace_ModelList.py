from huggingface_hub import scan_cache_dir

# Scan the local Hugging Face cache
cache_info = scan_cache_dir()

print("📦 Locally Cached Hugging Face Models:\n")
for repo in cache_info.repos:
    if repo.repo_type == "model":
        print(f"- Model ID: {repo.repo_id}")
        print(f"  Storage Size: {repo.size_on_disk_str}")
        print(f"  Path: {repo.repo_path}\n")