import streamlit as st
import pandas as pd
from service import preprocess_text, load_slang_dict

# Add a dynamic slang dictionary to the session state
if 'slang_dict' not in st.session_state:
    st.session_state.slang_dict = load_slang_dict()

st.title("Tomodachi Text Cleaning")
st.text("Kebersihan dataset adalah sebagian dari model yang bagus")
st.text("Hanya dengan beberapa klik dataset anda bersih")

# File upload
uploaded_file = st.file_uploader("Upload dataset anda CSV atau Excel", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    
    # Display the uploaded data
    st.write("Dataset anda:")
    st.write(df.head())
    
    # Select columns to clean
    columns_to_clean = st.multiselect("Pilih Kolom yang Ingin di Bersihkan", df.columns.tolist())
    
    if columns_to_clean:
        # Create a progress bar
        progress_bar = st.progress(0)
        
        # Create a status message
        status_message = st.empty()
        
        # Cleaning steps
        cleaning_steps = [
            'URL Removed', 
            'Lowercased', 
            'Tokenized', 
            'Punctuation Removed', 
            'Stopwords Removed', 
            'Stemmed',
            'Slang Normalisasi', 
        ]
        
        # Calculate total steps
        total_steps = len(columns_to_clean) * len(cleaning_steps)
        current_step = 0
        
        for col in columns_to_clean:
            for step in cleaning_steps:
                # Update status message
                status_message.text(f"Cleaning column '{col}': {step}")
                
                # Apply cleaning step
                if step == 'URL Removed':
                    df[f'{col}_url_removed'] = df[col].apply(lambda x: preprocess_text(x, st.session_state.slang_dict)['URL Removed'] if pd.notnull(x) else '')
                elif step == 'Lowercased':
                    df[f'{col}_lowercased'] = df[col].apply(lambda x: preprocess_text(x, st.session_state.slang_dict)['Lowercased'] if pd.notnull(x) else '')
                elif step == 'Tokenized':
                    df[f'{col}_tokenized'] = df[col].apply(lambda x: preprocess_text(x, st.session_state.slang_dict)['Tokenized'] if pd.notnull(x) else [])
                elif step == 'Punctuation Removed':
                    df[f'{col}_punctuation_removed'] = df[col].apply(lambda x: preprocess_text(x, st.session_state.slang_dict)['Punctuation Removed'] if pd.notnull(x) else '')
                elif step == 'Stopwords Removed':
                    df[f'{col}_stopwords_removed'] = df[col].apply(lambda x: preprocess_text(x, st.session_state.slang_dict)['Stopwords Removed'] if pd.notnull(x) else '')
                elif step == 'Stemmed':
                    df[f'{col}_stemmed'] = df[col].apply(lambda x: preprocess_text(x, st.session_state.slang_dict)['Stemmed'] if pd.notnull(x) else '')
                elif step == 'Slang Normalisasi':
                    df[f'{col}_slang_normalisasi'] = df[col].apply(lambda x: preprocess_text(x, st.session_state.slang_dict)['Slang Normalisasi'] if pd.notnull(x) else '')
                
                # Update progress
                current_step += 1
                progress_bar.progress(current_step / total_steps)
        
        # Clear status message and show completion
        status_message.text("Text Processing Selesai!")
        
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
        st.error("Anda belum memilih kolom!")

# Add slang input
st.subheader("Tambah Slang dan Formal")
st.write("Masukkan slang dan bentuk formal dalam format berikut:\n`slang1,formal1\nslang2,formal2`")

# Text area for multiple slang entries
slang_input = st.text_area("Masukkan Slang dan Bentuk Formal (format: slang,formal)", "")

if st.button("Tambah Slang"):
    if slang_input:
        new_entries = slang_input.strip().split('\n')
        added_slangs = []
        errors = []

        for entry in new_entries:
            parts = entry.split(',', 1)
            if len(parts) == 2:
                slang, formal = parts
                slang = slang.strip()
                formal = formal.strip()
                
                if slang and formal:
                    st.session_state.slang_dict[slang] = formal
                    added_slangs.append(f"{slang} -> {formal}")
                else:
                    errors.append(f"Invalid entry: {entry}")
            else:
                errors.append(f"Invalid format: {entry}")

        if added_slangs:
            st.success(f"Slang ditambahkan:\n" + "\n".join(added_slangs))
            # Re-run the processing after adding new slang
            if uploaded_file is not None and columns_to_clean:
                for col in columns_to_clean:
                    for step in cleaning_steps:
                        if step == 'Slang Normalisasi':
                            df[f'{col}_slang_normalisasi'] = df[col].apply(lambda x: preprocess_text(x, st.session_state.slang_dict)['Slang Normalisasi'] if pd.notnull(x) else '')
                
                st.write("Dataset Bersih setelah update slang:")
                st.write(df.head())
                
                st.download_button(
                    label="Download Dataset Bersih Sebagai CSV",
                    data=df.to_csv(index=False),
                    file_name="cleaned_data_updated.csv",
                    mime="text/csv",
                )
        if errors:
            st.error("Errors:\n" + "\n".join(errors))
    else:
        st.error("Input tidak boleh kosong")
