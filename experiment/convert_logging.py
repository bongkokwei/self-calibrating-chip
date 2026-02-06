#!/usr/bin/env python3
"""
Convert print() statements to logger.info() calls
"""
import re
from pathlib import Path
from datetime import datetime
import shutil


def backup_file(filepath):
    """Create timestamped backup"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = filepath.with_suffix(f".bak_{timestamp}{filepath.suffix}")
    shutil.copy2(filepath, backup_path)
    return backup_path


def needs_logger_import(content):
    """Check if file already imports logging"""
    return "import logging" not in content


def needs_logger_declaration(content):
    """Check if logger is already declared"""
    return not re.search(r"logger\s*=\s*logging\.getLogger", content)


def convert_prints(filepath, dry_run=False):
    """Convert print statements to logger.info in a file"""
    content = filepath.read_text(encoding="utf-8")
    original_content = content

    # Count print statements
    print_count = len(re.findall(r"\bprint\s*\(", content))
    if print_count == 0:
        return None

    # Convert print(...) to logger.info(...)
    content = re.sub(r"\bprint\s*\(", "logger.info(", content)

    # Add logging setup if needed
    lines = content.split("\n")
    insert_pos = 0

    # Find position after existing imports
    for i, line in enumerate(lines):
        if line.strip() and not line.strip().startswith("#"):
            if line.startswith("import ") or line.startswith("from "):
                insert_pos = i + 1
            else:
                break

    additions = []
    if needs_logger_import(original_content):
        additions.append("import logging")

    if needs_logger_declaration(original_content):
        additions.append("logger = logging.getLogger(__name__)")
        additions.append("")  # blank line

    if additions:
        lines = lines[:insert_pos] + additions + lines[insert_pos:]
        content = "\n".join(lines)

    if dry_run:
        print(f"  Would convert {print_count} print statement(s)")
        print(f"  Would add: {', '.join(a for a in additions if a)}")
        return None
    else:
        # Backup original
        backup_path = backup_file(filepath)
        # Write new content
        filepath.write_text(content, encoding="utf-8")
        return {
            "prints_converted": print_count,
            "backup": backup_path,
            "added_import": needs_logger_import(original_content),
            "added_logger": needs_logger_declaration(original_content),
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convert print() to logger.info()")
    parser.add_argument("path", help="Directory or file to process")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done"
    )
    parser.add_argument(
        "--pattern", default="*.py", help="File pattern (default: *.py)"
    )
    args = parser.parse_args()

    path = Path(args.path)

    if path.is_file():
        files = [path]
    else:
        files = sorted(path.rglob(args.pattern))

    print(f"{'DRY RUN: ' if args.dry_run else ''}Processing {len(files)} file(s)...")
    print()

    modified = []
    for filepath in files:
        print(f"📄 {filepath}")
        result = convert_prints(filepath, dry_run=args.dry_run)
        if result:
            modified.append((filepath, result))
            print(f"  ✅ Converted {result['prints_converted']} print(s)")
            print(f"  💾 Backup: {result['backup'].name}")
        elif result is None and not args.dry_run:
            print(f"  ⏭️  No print statements found")
        print()

    if modified and not args.dry_run:
        print(f"\n✨ Modified {len(modified)} file(s)")
        print(
            "\n⚠️  Don't forget to add logging configuration to your main entry point:"
        )
        print(
            "    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')"
        )


if __name__ == "__main__":
    main()
