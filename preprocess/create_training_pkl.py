#!/usr/bin/env python3
"""
Script to create training pkl file from processed CSV data.
Extracts relevant columns and creates a pandas DataFrame in pkl format for model training.
"""

import pandas as pd
import pickle
import argparse
import sys
from pathlib import Path

def create_training_pkl(input_csv: str, output_pkl: str = None,
                       id_column: str = 'transcript',
                       sequence_column: str = 'protein_sequence_cleaned',
                       label_column: str = 'protein_essentiality',
                       use_cleaned_sequence: bool = True,
                       write_fasta: bool = False,
                       fasta_output: str = None):
    """
    Create a training pkl file from processed CSV data.
    
    Args:
        input_csv: Path to input CSV file
        output_pkl: Path to output pkl file (default: input_csv with .pkl extension)
        id_column: Column name for sequence IDs
        sequence_column: Column name for protein sequences
        label_column: Column name for labels (human column)
        use_cleaned_sequence: Whether to use cleaned sequences (without stop codons)
    """
    print(f"Reading data from {input_csv}...")
    
    # Read CSV file
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows from CSV")
    
    # Check required columns
    required_columns = [id_column, label_column]
    
    # Determine sequence column to use - prioritize cleaned sequences
    if 'protein_sequence_cleaned' in df.columns and use_cleaned_sequence:
        sequence_col = 'protein_sequence_cleaned'
        print("Using cleaned protein sequences (stop codons removed)")
    elif 'protein_sequence_cleaned' in df.columns and not use_cleaned_sequence:
        sequence_col = 'protein_sequence'
        print("Using original protein sequences (cleaned sequences available but not requested)")
    elif 'protein_sequence' in df.columns:
        sequence_col = 'protein_sequence'
        print("Using original protein sequences (no cleaned sequences found)")
    else:
        raise ValueError("No protein sequence column found")
    
    required_columns.append(sequence_col)
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Available columns: {list(df.columns)}")
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Create training DataFrame with required structure
    training_df = pd.DataFrame()
    
    # Add index column (0-based sequential index)
    training_df['index'] = range(len(df))
    
    # Add ID column
    training_df['ID'] = df[id_column].astype(str)
    
    # Add label column (rename to 'human' as required by training code)
    training_df['human'] = df[label_column].astype(int)
    
    # Add sequence column
    training_df['sequence'] = df[sequence_col].astype(str)
    
    # Remove rows with NaN or empty sequences
    initial_count = len(training_df)
    training_df = training_df.dropna(subset=['sequence'])
    training_df = training_df[training_df['sequence'].str.strip() != '']
    
    removed_count = initial_count - len(training_df)
    if removed_count > 0:
        print(f"Removed {removed_count} rows with empty sequences")
    
    # Reset index to ensure continuous indexing
    training_df['index'] = range(len(training_df))
    training_df = training_df.reset_index(drop=True)
    
    # Print statistics
    print(f"\nTraining data statistics:")
    print(f"Total samples: {len(training_df)}")
    print(f"Label distribution:")
    label_counts = training_df['human'].value_counts().sort_index()
    for label, count in label_counts.items():
        percentage = count / len(training_df) * 100
        label_name = "essential" if label == 1 else "non-essential"
        print(f"  {label} ({label_name}): {count} ({percentage:.1f}%)")
    
    # Sequence length statistics
    seq_lengths = training_df['sequence'].str.len()
    print(f"\nSequence length statistics:")
    print(f"Min length: {seq_lengths.min()}")
    print(f"Max length: {seq_lengths.max()}")
    print(f"Mean length: {seq_lengths.mean():.2f}")
    print(f"Median length: {seq_lengths.median():.2f}")
    
    # Show sample data
    print(f"\nSample of training data:")
    print(training_df.head().to_string(index=False))
    
    if output_pkl is None:
        input_path = Path(input_csv)
        output_pkl = input_path.parent / "human_full.pkl"
    
    # Save pkl file
    print(f"\nSaving training data to {output_pkl}...")
    with open(output_pkl, 'wb') as f:
        pickle.dump(training_df, f)
    
    print(f"Training pkl file created successfully!")
    print(f"File size: {Path(output_pkl).stat().st_size / 1024 / 1024:.2f} MB")
    
    # Verify the saved file
    print(f"\nVerifying saved file...")
    with open(output_pkl, 'rb') as f:
        loaded_df = pickle.load(f)
    
    print(f"Verification successful - loaded {len(loaded_df)} rows")
    print(f"Columns: {list(loaded_df.columns)}")
    
    if write_fasta:
        if fasta_output is None:
            fasta_output = str(Path(input_csv).parent / "human_full.fasta")
        print(f"\nWriting FASTA to {fasta_output}...")
        with open(fasta_output, 'w') as fout:
            for _, row in training_df.iterrows():
                fout.write(f">index-{row['index']}\n")
                fout.write(f"{row['sequence']}\n")
        print("FASTA writing completed")

    return training_df

def main():
    parser = argparse.ArgumentParser(description='Create training pkl file from processed CSV data')
    parser.add_argument('input_csv', help='Path to input CSV file (relative to project root or absolute path)')
    parser.add_argument('-o', '--output', help='Output pkl file path')
    parser.add_argument('--id-column', default='transcript', help='Column name for sequence IDs (default: transcript)')
    parser.add_argument('--sequence-column', default='protein_sequence_cleaned', help='Column name for sequences (default: protein_sequence_cleaned)')
    parser.add_argument('--label-column', default='protein_essentiality', help='Column name for labels (default: protein_essentiality)')
    parser.add_argument('--use-original-sequence', action='store_true', help='Use original sequences instead of cleaned sequences')
    parser.add_argument('--preview-only', action='store_true', help='Only preview the data without creating pkl file')
    parser.add_argument('--write-fasta', action='store_true', help='Also export FASTA with header index-<index>')
    parser.add_argument('--fasta-output', help='Output FASTA path')
    
    args = parser.parse_args()
    
    try:
        if args.preview_only:
            print("PREVIEW MODE: Analyzing data without creating pkl file")
            df = pd.read_csv(args.input_csv)
            print(f"CSV file contains {len(df)} rows")
            print(f"Available columns: {list(df.columns)}")
            
            # Check for required columns
            if args.label_column in df.columns:
                label_counts = df[args.label_column].value_counts()
                print(f"Label distribution ({args.label_column}):")
                for label, count in label_counts.items():
                    print(f"  {label}: {count}")
            
            if args.sequence_column in df.columns:
                seq_lengths = df[args.sequence_column].str.len()
                print(f"Sequence lengths ({args.sequence_column}):")
                print(f"  Min: {seq_lengths.min()}")
                print(f"  Max: {seq_lengths.max()}")
                print(f"  Mean: {seq_lengths.mean():.2f}")
            elif 'protein_sequence' in df.columns:
                seq_lengths = df['protein_sequence'].str.len()
                print(f"Sequence lengths (protein_sequence):")
                print(f"  Min: {seq_lengths.min()}")
                print(f"  Max: {seq_lengths.max()}")
                print(f"  Mean: {seq_lengths.mean():.2f}")
            
            return
        
        use_cleaned = not args.use_original_sequence
        
        create_training_pkl(
            args.input_csv,
            args.output,
            args.id_column,
            args.sequence_column,
            args.label_column,
            use_cleaned,
            args.write_fasta,
            args.fasta_output
        )
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
