import streamlit as st
import pandas as pd
from service import preprocess_text

st.title("Text Processing Web App")
st.text("Kebersihan dataset adalah sebagian dari model yang bagus")
st.text("Hanya dengan beberapa klik dataset anda bersih")

# File upload
uploaded_file = st.file_uploader("upload dataset anda CSv atau Excel", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    
    # Display the uploaded data
    st.write("dataset anda:")
    st.write(df.head())
    
    # Select columns to clean
    columns_to_clean = st.multiselect("Pilih Kolom yang Ingin di Bersihkan", df.columns.tolist())
    
    if columns_to_clean:
        # Create a progress bar
        progress_bar = st.progress(0)
        
        # Create a status message
        status_message = st.empty()
        
        # Cleaning steps
        cleaning_steps = ['Lowercase', 'Tokenize', 'Remove Punctuation', 'Remove Stopwords', 'Stem']
        
        # Calculate total steps
        total_steps = len(columns_to_clean) * len(cleaning_steps)
        current_step = 0
        
        for col in columns_to_clean:
            for step in cleaning_steps:
                # Update status message
                status_message.text(f"Cleaning column '{col}': {step}")
                
                # Apply cleaning step
                if step == 'Lowercase':
                    df[f'{col}_lowercased'] = df[col].apply(lambda x: preprocess_text(x)['Lowercased'] if pd.notnull(x) else '')
                elif step == 'Tokenize':
                    df[f'{col}_tokenized'] = df[col].apply(lambda x: preprocess_text(x)['Tokenized'] if pd.notnull(x) else [])
                elif step == 'Remove Punctuation':
                    df[f'{col}_punctuation_removed'] = df[col].apply(lambda x: preprocess_text(x)['Punctuation Removed'] if pd.notnull(x) else '')
                elif step == 'Remove Stopwords':
                    df[f'{col}_stopwords_removed'] = df[col].apply(lambda x: preprocess_text(x)['Stopwords Removed'] if pd.notnull(x) else '')
                elif step == 'Stem':
                    df[f'{col}_stemmed'] = df[col].apply(lambda x: preprocess_text(x)['Stemmed'] if pd.notnull(x) else '')
                
                # Update progress
                current_step += 1
                progress_bar.progress(current_step / total_steps)
        
        # Clear status message and show completion
        status_message.text("text Processing Selesai!")
        
        st.write("Dataset Bersih:")
        st.write(df.head())
        
        # Option to download cleaned data
        st.download_button(
            label="Download Dataset Bersih Sebagai CSV",
            data=df.to_csv(index=False),
            file_name="cleaned_data.csv",
            mime="text/csv",
        )
    else:
        st.error("tAnda belum memilih kolom!.")