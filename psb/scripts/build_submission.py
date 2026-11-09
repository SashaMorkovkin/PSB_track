"""Собирает ZIP-сабмит: обязательные файлы находятся в корне архива."""
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/"submission.zip"

def main():
    required=[ROOT/"solution.py",ROOT/"requirements.txt"]
    missing=[str(p) for p in required if not p.is_file()]
    if missing: raise FileNotFoundError(", ".join(missing))
    with zipfile.ZipFile(OUTPUT,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for p in required: z.write(p,arcname=p.name)
    print(f"Создан {OUTPUT} ({OUTPUT.stat().st_size} байт)")

if __name__=="__main__": main()
