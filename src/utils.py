import json
import os
from pathlib import Path
from typing import Any, List, Dict, Union, Optional

import pandas as pd


# Default encoding for Vietnamese text
VIETNAMESE_ENCODING = "utf-8"


def read_json(file_path: str) -> Any:
    """
    Read JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Parsed JSON data
    """
    with open(file_path, "r", encoding=VIETNAMESE_ENCODING) as f:
        return json.load(f)


def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Read JSONL (JSON Lines) file.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of parsed JSON objects
    """
    data = []
    with open(file_path, "r", encoding=VIETNAMESE_ENCODING) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line: 
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"JSONL syntax error at line {line_num}: {line[:50]}...")
    return data


def read_parquet(file_path: str) -> List[Dict[str, Any]]:
    """
    Read Parquet file.
    
    Args:
        file_path: Path to Parquet file
        
    Returns:
        List of records as dictionaries
    """
    df = pd.read_parquet(file_path)
    return df.to_dict(orient="records")


def read_csv(file_path: str, **kwargs) -> List[Dict[str, Any]]:
    """
    Read CSV file.
    
    Args:
        file_path: Path to CSV file
        **kwargs: Additional arguments for pd.read_csv
        
    Returns:
        List of records as dictionaries
    """
    kwargs.setdefault("encoding", VIETNAMESE_ENCODING)
    df = pd.read_csv(file_path, **kwargs)
    return df.to_dict(orient="records")


def read_txt(file_path: str) -> List[str]:
    """
    Read text file.
    
    Args:
        file_path: Path to text file
        
    Returns:
        List of lines with whitespace stripped
    """
    with open(file_path, "r", encoding=VIETNAMESE_ENCODING) as f:
        return [line.strip() for line in f if line.strip()]


def save_json(
    file_path: str, 
    obj: Any, 
    verbose: bool = True
) -> None:
    """
    Save object as JSON file.
    
    Args:
        file_path: Path to save JSON file
        obj: Object to save
        verbose: Whether to print status message
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding=VIETNAMESE_ENCODING) as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)
    if verbose: 
        print(f"JSON saved: {file_path}")


def save_jsonl(
    file_path: str, 
    obj: List[Any], 
    verbose: bool = True
) -> None:
    """
    Save list of objects as JSONL file.
    
    Args:
        file_path: Path to save JSONL file
        obj: List of objects to save
        verbose: Whether to print status message
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding=VIETNAMESE_ENCODING) as f:
        for item in obj:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    if verbose: 
        print(f"JSONL saved: {file_path} | Records: {len(obj)}")


def save_parquet(
    file_path: str, 
    obj: List[Dict[str, Any]], 
    engine: str = "pyarrow", 
    verbose: bool = True
) -> None:
    """
    Save list of objects as Parquet file.
    
    Args:
        file_path: Path to save Parquet file
        obj: List of dictionaries to save
        engine: Parquet engine to use
        verbose: Whether to print status message
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(obj)
    df.to_parquet(file_path, engine=engine, index=False)
    if verbose: 
        print(f"Parquet saved: {file_path} | Records: {len(df)}")