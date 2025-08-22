"""
Quick script to find and fix potential view() issues in the codebase
"""

import os
import re
from pathlib import Path

def find_view_calls(directory):
    """Find all .view() calls in Python files"""
    view_calls = []
    
    for file_path in Path(directory).rglob("*.py"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
                for i, line in enumerate(lines):
                    if '.view(' in line and 'reshape' not in line:
                        view_calls.append({
                            'file': str(file_path),
                            'line_num': i + 1,
                            'line': line.strip()
                        })
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    return view_calls

def main():
    print("🔍 Searching for potential .view() issues...")
    
    # Search in models directory
    view_calls = find_view_calls("models/")
    
    if view_calls:
        print(f"\n⚠️  Found {len(view_calls)} .view() calls:")
        for call in view_calls:
            print(f"📁 {call['file']}:{call['line_num']}")
            print(f"   {call['line']}")
            print()
        
        print("💡 Consider replacing .view() with .reshape() or .contiguous().view()")
    else:
        print("✅ No .view() calls found!")

if __name__ == "__main__":
    main()
