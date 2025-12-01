import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima.arima import auto_arima
import openpyxl
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="FORRISX - Forecasting & Risk Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS dengan dark mode yang nyaman
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Dark background gradient */
    .main {
        background: linear-gradient(-45deg, #0f0f1e, #1a1a2e, #16213e, #0f3460);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Sidebar dark */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a15 0%, #1a1a2e 100%);
        border-right: 2px solid rgba(102, 126, 234, 0.3);
    }
    
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    
    /* Card containers */
    div[data-testid="stVerticalBlock"] > div {
        background: rgba(26, 26, 46, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 700 !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 800;
        color: #ffffff !important;
        text-shadow: 0 0 10px rgba(102, 126, 234, 0.5);
    }
    
    [data-testid="stMetricLabel"] {
        color: #b0b0c0 !important;
        font-weight: 600;
    }
    
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15), rgba(118, 75, 162, 0.15));
        border-radius: 12px;
        padding: 18px;
        border: 1px solid rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
        border-color: rgba(102, 126, 234, 0.5);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 30px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
    }
    
    /* Radio & Input */
    .stRadio > label, .stSelectbox > label, .stNumberInput > label {
        color: #e0e0e0 !important;
        font-weight: 600;
    }
    
    .stRadio > div {
        background: rgba(26, 26, 46, 0.5);
        padding: 10px;
        border-radius: 10px;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(26, 26, 46, 0.5);
        border: 2px dashed rgba(102, 126, 234, 0.4);
        border-radius: 12px;
        padding: 25px;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(102, 126, 234, 0.7);
        background: rgba(26, 26, 46, 0.7);
    }
    
    /* Input fields */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background: rgba(26, 26, 46, 0.8) !important;
        color: #ffffff !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 8px !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(26, 26, 46, 0.5);
        padding: 8px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(26, 26, 46, 0.7);
        border-radius: 8px;
        padding: 10px 20px;
        color: #b0b0c0;
        font-weight: 600;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: #ffffff;
        border-color: rgba(102, 126, 234, 0.5);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(26, 26, 46, 0.7);
        border-radius: 10px;
        color: #ffffff !important;
        font-weight: 600;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    /* Alert boxes */
    .stAlert {
        background: rgba(26, 26, 46, 0.8) !important;
        border-radius: 10px;
        border-left: 4px solid;
    }
    
    /* Text */
    p, span, label, div, li {
        color: #d0d0e0 !important;
    }
    
    /* Dataframe */
    .dataframe {
        background: rgba(26, 26, 46, 0.5) !important;
        border-radius: 10px;
    }
    
    .dataframe th {
        background: rgba(102, 126, 234, 0.3) !important;
        color: #ffffff !important;
        font-weight: 600;
    }
    
    .dataframe td {
        background: rgba(26, 26, 46, 0.5) !important;
        color: #e0e0e0 !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(26, 26, 46, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #667eea, #764ba2);
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 20px 0;'>
            <h1 style='font-size: 2.5rem; margin: 0;'>📈</h1>
            <h2 style='margin: 10px 0; font-size: 1.4rem;'>FORRISX</h2>
            <p style='opacity: 0.7; font-size: 0.85rem; line-height: 1.5;'>
                Forecasting & Risk Analysis<br>System
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr style='border: 1px solid rgba(102,126,234,0.3); margin: 15px 0;'>", unsafe_allow_html=True)
    
    page = st.radio(
        "Navigation",
        ["🏠 Home", "📁 Upload & EDA", "📊 Data Visualization", "🤖 ARIMAX Modeling", "🔮 Forecasting", "⚠️ Value-at-Risk"],
        label_visibility="visible"
    )
    
    st.markdown("<hr style='border: 1px solid rgba(102,126,234,0.3); margin: 15px 0;'>", unsafe_allow_html=True)
    
    st.markdown("### 📚 Guide")
    steps = [
        "1️⃣ Learn about methods",
        "2️⃣ Upload your data",
        "3️⃣ Visualize trends",
        "4️⃣ Build ARIMAX model",
        "5️⃣ Generate forecast",
        "6️⃣ Calculate risk"
    ]
    for step in steps:
        st.markdown(f"<div style='padding: 5px; font-size: 0.85rem;'>{step}</div>", unsafe_allow_html=True)

# ========== PAGE 1: HOME ==========
if page == "🏠 Home":
    st.markdown("""
        <div style='text-align: center; padding: 40px 20px;'>
            <h1 style='font-size: 3.8rem; margin-bottom: 5px; letter-spacing: 3px;'>
                FORRISX
            </h1>
            <p style='font-size: 1.2rem; opacity: 0.85; margin-bottom: 10px; letter-spacing: 1px;'>
                Forecasting and Risk Analysis System using ARIMAX and VaR
            </p>
            <p style='font-size: 1rem; opacity: 0.7; margin-bottom: 40px;'>
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div style='background: rgba(102, 126, 234, 0.1); padding: 25px; border-radius: 15px; 
                        border-left: 4px solid #667eea; margin: 10px 0;'>
                <h3 style='color: #667eea; margin-top: 0;'>🤖 ARIMAX Method</h3>
                <p style='line-height: 1.8; font-size: 0.95rem;'>
                    <strong>ARIMAX (AutoRegressive Integrated Moving Average with eXogenous variables)</strong> 
                    adalah model statistik yang digunakan untuk prediksi time series. Model ini menggabungkan 
                    tiga komponen utama:
                </p>
                <ul style='line-height: 1.8; font-size: 0.95rem;'>
                    <li><strong>AR (AutoRegressive):</strong> Menggunakan nilai masa lalu untuk prediksi</li>
                    <li><strong>I (Integrated):</strong> Melakukan differencing untuk stasioneritas data</li>
                    <li><strong>MA (Moving Average):</strong> Menggunakan error masa lalu</li>
                    <li><strong>X (eXogenous):</strong> Memasukkan variabel eksternal (USD, SGD)</li>
                </ul>
                <p style='line-height: 1.8; font-size: 0.95rem;'>
                    Model ini cocok untuk memprediksi harga saham dengan mempertimbangkan pengaruh 
                    nilai tukar mata uang asing.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='background: rgba(240, 147, 251, 0.1); padding: 25px; border-radius: 15px; 
                        border-left: 4px solid #f093fb; margin: 10px 0;'>
                <h3 style='color: #f093fb; margin-top: 0;'>⚠️ Value-at-Risk (VaR)</h3>
                <p style='line-height: 1.8; font-size: 0.95rem;'>
                    <strong>Value-at-Risk (VaR)</strong> adalah metrik risiko yang mengukur potensi 
                    kerugian maksimum dari suatu investasi dalam periode waktu tertentu dengan tingkat 
                    kepercayaan tertentu.
                </p>
                <ul style='line-height: 1.8; font-size: 0.95rem;'>
                    <li><strong>Confidence Level:</strong> Tingkat kepercayaan (95% atau 99%)</li>
                    <li><strong>Time Horizon:</strong> Periode waktu investasi</li>
                    <li><strong>Log Returns:</strong> Perubahan harga dalam bentuk logaritma</li>
                </ul>
                <p style='line-height: 1.8; font-size: 0.95rem;'>
                    Contoh: VaR 95% sebesar 5% artinya ada kemungkinan 5% kerugian akan melebihi 5% 
                    dari nilai investasi dalam periode yang ditentukan. Metrik ini membantu investor 
                    dalam manajemen risiko portofolio.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Author info
    st.markdown("""
        <div style='text-align: center; padding: 30px; background: rgba(67, 233, 123, 0.1); 
                    border-radius: 15px; border: 1px solid rgba(67, 233, 123, 0.3); margin-top: 40px;'>
            <h3 style='color: #43e97b; margin-bottom: 15px;'>👨‍💻 Created By</h3>
            <h2 style='font-size: 2rem; margin: 10px 0;'>Amanda Aulia</h2>
            <p style='font-size: 1.2rem; opacity: 0.9;'>Sains Data 2021</p>
            <p style='font-size: 0.9rem; opacity: 0.7; margin-top: 15px;'>
                📊 Data Science | 📈 Financial Analytics | 🤖 Machine Learning
            </p>
        </div>
    """, unsafe_allow_html=True)

# ========== PAGE 2: UPLOAD & EDA ==========
elif page == "📁 Upload & EDA":
    st.markdown("<h2 style='text-align: center;'>📁 Data Upload & Exploratory Data Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; opacity: 0.8; margin-bottom: 30px;'>Upload your financial data and explore key statistics</p>", unsafe_allow_html=True)
    
    # Upload section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded_file = st.file_uploader(
            "📂 Upload your data file",
            type=['csv', 'xlsx', 'xls'],
            help="Supported: CSV, Excel"
        )
    
    if uploaded_file is not None:
        with st.spinner("🔄 Processing data..."):
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            
            st.session_state['df'] = df
        
        st.success("✅ Data uploaded successfully!", icon="🎉")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Data Information
        st.markdown("### 📊 Data Information")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("📝 Total Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("📋 Total Columns", f"{df.shape[1]}")
        with col3:
            st.metric("⚠️ Missing Values", f"{df.isnull().sum().sum()}")
        with col4:
            st.metric("🔄 Duplicate Rows", f"{df.duplicated().sum()}")
        with col5:
            st.metric("💾 Memory", f"{df.memory_usage().sum() / 1024:.1f} KB")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Data Types
        with st.expander("🔍 Data Types & Details", expanded=False):
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Non-Null': df.count(),
                'Null': df.isnull().sum(),
                'Unique': df.nunique()
            })
            st.dataframe(col_info, use_container_width=True, height=300)
        
        # Preview
        with st.expander("👀 Data Preview", expanded=True):
            st.dataframe(df.head(10), use_container_width=True, height=350)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Descriptive Statistics
        st.markdown("### 📈 Descriptive Statistics")
        
        cols_to_describe = [col for col in ['bbca', 'usd', 'sgd'] if col in df.columns]
        if cols_to_describe:
            desc_df = df[cols_to_describe].describe()
            st.dataframe(
                desc_df.style.background_gradient(cmap='viridis', axis=1).format("{:.2f}"),
                use_container_width=True
            )
            
            # Summary metrics
            st.markdown("#### 📊 Key Metrics")
            cols = st.columns(len(cols_to_describe))
            for idx, col_name in enumerate(cols_to_describe):
                with cols[idx]:
                    mean_val = df[col_name].mean()
                    std_val = df[col_name].std()
                    st.metric(
                        f"{col_name.upper()} Mean",
                        f"{mean_val:.2f}",
                        delta=f"σ: {std_val:.2f}"
                    )
        else:
            st.warning("Columns 'bbca', 'usd', 'sgd' not found in data")

# ========== PAGE 3: DATA VISUALIZATION ==========
elif page == "📊 Data Visualization":
    st.markdown("<h2 style='text-align: center;'>📊 Data Visualization & Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; opacity: 0.8; margin-bottom: 30px;'>Visualize trends, correlations, and outliers</p>", unsafe_allow_html=True)
    
    if 'df' not in st.session_state:
        st.warning("⚠️ Please upload data first", icon="📁")
    else:
        df = st.session_state['df']
        
        # Stock Price Visualization
        st.markdown("### 📈 Stock Price & Exchange Rate Trends")
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
            fig, ax = plt.subplots(figsize=(16, 7))
            fig.patch.set_facecolor('#0f0f19')
            ax.set_facecolor('#1a1a2e')
            
            colors = ['#667eea', '#f093fb', '#4facfe']
            
            if 'bbca' in df.columns:
                ax.plot(df['date'], df['bbca'], label='BBCA Stock', linewidth=2.5, color=colors[0], alpha=0.9)
            if 'usd' in df.columns:
                ax.plot(df['date'], df['usd'], label='USD/IDR', linewidth=2.5, color=colors[1], alpha=0.9)
            if 'sgd' in df.columns:
                ax.plot(df['date'], df['sgd'], label='SGD/IDR', linewidth=2.5, color=colors[2], alpha=0.9)
            
            ax.set_xlabel('Date', fontsize=13, color='white', fontweight='bold')
            ax.set_ylabel('Value', fontsize=13, color='white', fontweight='bold')
            ax.set_title('Time Series Analysis', fontsize=17, color='white', fontweight='bold', pad=20)
            ax.legend(loc='upper left', fontsize=12, framealpha=0.8, facecolor='#1a1a2e', edgecolor='#667eea')
            ax.grid(True, alpha=0.2, linestyle='--', color='white')
            ax.tick_params(colors='white', labelsize=11)
            
            for spine in ax.spines.values():
                spine.set_edgecolor('#667eea')
                spine.set_alpha(0.5)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Correlation Matrix
        st.markdown("### 🔗 Correlation Analysis")
        
        if all(col in df.columns for col in ['bbca', 'usd', 'sgd']):
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                fig, ax = plt.subplots(figsize=(9, 7))
                fig.patch.set_facecolor('#0f0f19')
                ax.set_facecolor('#1a1a2e')
                
                corr_matrix = df[['bbca', 'usd', 'sgd']].corr()
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                
                sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', fmt=".3f", 
                           linewidths=3, square=True, cbar_kws={"shrink": 0.8},
                           annot_kws={"size": 15, "weight": "bold", "color": "white"},
                           mask=mask, vmin=-1, vmax=1)
                
                ax.set_title('Correlation Heatmap', fontsize=17, color='white', fontweight='bold', pad=20)
                plt.xticks(rotation=0, ha='center', color='white', fontsize=12, fontweight='bold')
                plt.yticks(rotation=0, color='white', fontsize=12, fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Correlation insights
            st.markdown("#### 📌 Correlation Insights")
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    col1_name = corr_matrix.columns[i].upper()
                    col2_name = corr_matrix.columns[j].upper()
                    
                    if abs(corr_val) > 0.7:
                        st.error(f"🔴 **Strong correlation:** {col1_name} ↔ {col2_name} = **{corr_val:.3f}**")
                    elif abs(corr_val) > 0.4:
                        st.info(f"🟡 **Moderate correlation:** {col1_name} ↔ {col2_name} = **{corr_val:.3f}**")
                    else:
                        st.success(f"🟢 **Weak correlation:** {col1_name} ↔ {col2_name} = **{corr_val:.3f}**")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Boxplot for Outliers
        st.markdown("### 📦 Outlier Detection (Boxplot)")
        
        cols_to_plot = [col for col in ['bbca', 'usd', 'sgd'] if col in df.columns]
        if cols_to_plot:
            fig, axes = plt.subplots(1, len(cols_to_plot), figsize=(15, 6))
            fig.patch.set_facecolor('#0f0f19')
            
            if len(cols_to_plot) == 1:
                axes = [axes]
            
            colors_box = ['#667eea', '#f093fb', '#4facfe']
            
            for idx, col in enumerate(cols_to_plot):
                box = axes[idx].boxplot(df[col].dropna(), patch_artist=True,
                                       boxprops=dict(facecolor=colors_box[idx], alpha=0.7),
                                       medianprops=dict(color='white', linewidth=2),
                                       whiskerprops=dict(color='white'),
                                       capprops=dict(color='white'),
                                       flierprops=dict(marker='o', markerfacecolor='red', markersize=8, alpha=0.7))
                
                axes[idx].set_title(f'{col.upper()} Outliers', color='white', fontweight='bold', fontsize=14)
                axes[idx].set_ylabel('Value', color='white', fontweight='bold')
                axes[idx].set_facecolor('#1a1a2e')
                axes[idx].tick_params(colors='white')
                axes[idx].grid(True, alpha=0.2, color='white', axis='y')
                
                for spine in axes[idx].spines.values():
                    spine.set_edgecolor('#667eea')
                    spine.set_alpha(0.5)
                
                # Count outliers
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)][col].count()
                axes[idx].text(0.5, 0.95, f'Outliers: {outliers}', 
                              transform=axes[idx].transAxes, ha='center', va='top',
                              color='#ff6b6b', fontweight='bold', fontsize=11,
                              bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.8))
            
            plt.tight_layout()
            st.pyplot(fig)

# ========== PAGE 4: ARIMAX MODELING ==========
elif page == "🤖 ARIMAX Modeling":
    st.markdown("<h2 style='text-align: center;'>🤖 ARIMAX Model Development</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; opacity: 0.8; margin-bottom: 30px;'>Build and train your forecasting model</p>", unsafe_allow_html=True)
    
    if 'df' not in st.session_state:
        st.warning("⚠️ Please upload data first", icon="📁")
    else:
        df = st.session_state['df']
        
        # Split Data
        st.markdown("### 🔪 Data Splitting")
        
        col1, col2 = st.columns(2)
        with col1:
            split_method = st.radio(
                "Choose split method:",
                ["📊 Manual (Slider)", "🎯 Fixed (Index 1386)"],
                horizontal=True
            )
        
        with col2:
            if split_method == "📊 Manual (Slider)":
                split_ratio = st.slider("Training data ratio (%)", 50, 95, 80)
                split_index = int(len(df) * split_ratio / 100)
            else:
                split_index = 1386
                if split_index > len(df):
                    st.error(f"❌ Index exceeds data length ({len(df)})")
                    split_index = int(len(df) * 0.8)
                else:
                    st.success(f"✅ Split at index {split_index}")
        
        train = df[:split_index]
        test = df[split_index:]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📚 Training", f"{train.shape[0]:,}", f"{split_index/len(df)*100:.1f}%")
        with col2:
            st.metric("🧪 Testing", f"{test.shape[0]:,}", f"{(len(df)-split_index)/len(df)*100:.1f}%")
        with col3:
            st.metric("📊 Total", f"{len(df):,}")
        
        train_bbca = train['bbca']
        test_bbca = test['bbca']
        x_train = train[['usd', 'sgd']]
        x_test = test[['usd', 'sgd']]
        
        st.session_state.update({
            'train_bbca': train_bbca,
            'test_bbca': test_bbca,
            'x_train': x_train,
            'x_test': x_test,
            'train': train,
            'test': test
        })
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Auto ARIMA
        st.markdown("### 🔍 Determine Optimal ARIMA Order")
        st.info("💡 Use auto_arima to find the best (p,d,q) parameters automatically", icon="💡")
        
        if st.button("🚀 Run Auto ARIMA", use_container_width=False):
            with st.spinner("🔄 Finding optimal parameters... This may take a few minutes"):
                try:
                    model_auto = auto_arima(
                        train_bbca,
                        seasonal=False,
                        start_p=1,
                        d=1,
                        start_q=1,
                        max_p=10,
                        max_d=1,
                        max_q=10,
                        stepwise=True,
                        suppress_warnings=True,
                        trace=True
                    )
                    
                    st.success("✅ Optimal parameters found!", icon="🎉")
                    
                    # Display results
                    st.markdown("#### 📊 Auto ARIMA Results")
                    order = model_auto.order
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("📈 p (AR)", order[0])
                    with col2:
                        st.metric("📉 d (I)", order[1])
                    with col3:
                        st.metric("📊 q (MA)", order[2])
                    with col4:
                        st.metric("🎯 AIC", f"{model_auto.aic():.2f}")
                    with col5:
                        st.metric("🎯 BIC", f"{model_auto.bic():.2f}")
                    
                    with st.expander("📋 Detailed Summary", expanded=False):
                        st.text(str(model_auto.summary()))
                    
                    # Save to session
                    st.session_state['auto_order'] = order
                    st.info(f"💡 Recommended order: ({order[0]}, {order[1]}, {order[2]})", icon="✨")
                    
                except Exception as e:
                    st.error(f"❌ Auto ARIMA failed: {str(e)}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Manual ARIMAX
        st.markdown("### ⚙️ Build ARIMAX Model")
        st.info("💡 Enter parameters manually or use the recommended values from Auto ARIMA above", icon="💡")
        
        col1, col2, col3 = st.columns(3)
        
        # Set default values from auto_arima if available
        default_p = st.session_state.get('auto_order', (0, 1, 1))[0]
        default_d = st.session_state.get('auto_order', (0, 1, 1))[1]
        default_q = st.session_state.get('auto_order', (0, 1, 1))[2]
        
        with col1:
            p = st.number_input("**p** (AR order)", 0, 10, int(default_p), 
                               help="Autoregressive order")
        with col2:
            d = st.number_input("**d** (Differencing)", 0, 2, int(default_d),
                               help="Degree of differencing")
        with col3:
            q = st.number_input("**q** (MA order)", 0, 10, int(default_q),
                               help="Moving average order")
        
        st.session_state.update({'p': p, 'd': d, 'q': q})
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 BUILD ARIMAX MODEL", use_container_width=True):
                progress = st.progress(0)
                status = st.empty()
                
                for i in range(100):
                    progress.progress(i + 1)
                    if i < 30:
                        status.text("🔄 Initializing...")
                    elif i < 70:
                        status.text("🧠 Training model...")
                    else:
                        status.text("✨ Finalizing...")
                
                try:
                    model = SARIMAX(endog=train_bbca, exog=x_train, order=(p, d, q))
                    results = model.fit(disp=False)
                    st.session_state['model_results'] = results
                    
                    progress.empty()
                    status.empty()
                    
                    st.balloons()
                    st.success("✅ Model trained successfully!", icon="🎉")
                    
                    st.markdown("#### 📊 Model Performance")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("AIC", f"{results.aic:.2f}", help="Lower is better")
                    with col2:
                        st.metric("BIC", f"{results.bic:.2f}", help="Lower is better")
                    with col3:
                        st.metric("Log-Likelihood", f"{results.llf:.2f}")
                    
                    with st.expander("📋 Model Summary", expanded=False):
                        st.text(str(results.summary()))
                    
                except Exception as e:
                    progress.empty()
                    status.empty()
                    st.error(f"❌ Failed: {str(e)}")

# ========== PAGE 5: FORECASTING ==========
elif page == "🔮 Forecasting":
    st.markdown("<h2 style='text-align: center;'>🔮 Stock Price Forecasting</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; opacity: 0.8; margin-bottom: 30px;'>Generate predictions and evaluate model accuracy</p>", unsafe_allow_html=True)
    
    if 'model_results' not in st.session_state:
        st.warning("⚠️ Please build ARIMAX model first", icon="🤖")
    else:
        results = st.session_state['model_results']
        test_bbca = st.session_state['test_bbca']
        x_test = st.session_state['x_test']
        train_bbca = st.session_state['train_bbca']
        test = st.session_state['test']
        
        with st.spinner("🔮 Generating forecasts..."):
            forecast_bbca = results.get_forecast(steps=len(test_bbca), exog=x_test)
            forecast_bbca_values = forecast_bbca.predicted_mean
            conf_int_bbca = forecast_bbca.conf_int()
        
        # Visualization
        st.markdown("### 📈 Forecast Visualization")
        
        fig, ax = plt.subplots(figsize=(18, 8))
        fig.patch.set_facecolor('#0f0f19')
        ax.set_facecolor('#1a1a2e')
        
        ax.plot(train_bbca.index, train_bbca, label='Training Data', 
                color='#4facfe', linewidth=2.5, alpha=0.9)
        ax.plot(test_bbca.index, test_bbca, label='Actual Test', 
                color='#f093fb', linewidth=2.5, alpha=0.9)
        ax.plot(test_bbca.index, forecast_bbca_values, label='Forecast', 
                color='#43e97b', linewidth=3, linestyle='--', alpha=0.95)
        ax.fill_between(test_bbca.index, conf_int_bbca.iloc[:, 0], conf_int_bbca.iloc[:, 1], 
                        color='#43e97b', alpha=0.2, label='95% Confidence Interval')
        
        ax.set_xlabel('Index', fontsize=13, color='white', fontweight='bold')
        ax.set_ylabel('BBCA Price (IDR)', fontsize=13, color='white', fontweight='bold')
        ax.set_title('Stock Price Forecast with Confidence Interval', 
                    fontsize=18, color='white', fontweight='bold', pad=20)
        ax.legend(loc='upper left', fontsize=12, framealpha=0.8, facecolor='#1a1a2e', edgecolor='#667eea')
        ax.grid(True, alpha=0.2, linestyle='--', color='white')
        ax.tick_params(colors='white', labelsize=11)
        
        for spine in ax.spines.values():
            spine.set_edgecolor('#667eea')
            spine.set_alpha(0.5)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Comparison Table
        st.markdown("### 📊 Forecast vs Actual Comparison")
        
        df_forecast = pd.DataFrame({
            'Date': test['date'].values if 'date' in test.columns else test.index,
            'Actual': test['bbca'].values,
            'Forecast': forecast_bbca_values.values,
            'Difference': test['bbca'].values - forecast_bbca_values.values,
            'Error (%)': np.abs((test['bbca'].values - forecast_bbca_values.values) / test['bbca'].values * 100)
        })
        
        st.dataframe(
            df_forecast.style.background_gradient(subset=['Error (%)'], cmap='RdYlGn_r').format({
                'Actual': '{:.2f}',
                'Forecast': '{:.2f}',
                'Difference': '{:.2f}',
                'Error (%)': '{:.2f}%'
            }),
            use_container_width=True,
            height=400
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # MAPE Calculation
        st.markdown("### 🎯 Model Accuracy (MAPE)")
        
        actual = test['bbca'].values
        forecast = forecast_bbca_values.values
        mape = np.mean(np.abs((actual - forecast) / actual)) * 100
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.metric(
                "MAPE (Mean Absolute Percentage Error)",
                f"{mape:.2f}%",
                delta=f"{'Excellent' if mape < 10 else 'Good' if mape < 20 else 'Needs Work'}"
            )
            
            if mape < 10:
                st.success(
                    "🌟 **Excellent Performance!** MAPE < 10% indicates high accuracy. "
                    "This model is highly reliable for forecasting.", 
                    icon="✅"
                )
            elif mape < 20:
                st.info(
                    "👍 **Good Performance!** MAPE < 20% shows satisfactory accuracy. "
                    "The model can be used for forecasting with reasonable confidence.", 
                    icon="ℹ️"
                )
            else:
                st.warning(
                    "⚠️ **Needs Improvement!** MAPE ≥ 20% suggests lower accuracy. "
                    "Consider adjusting parameters or adding more features.", 
                    icon="🔧"
                )
        
        # Additional metrics
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📊 Additional Metrics")
        
        mae = np.mean(np.abs(actual - forecast))
        rmse = np.sqrt(np.mean((actual - forecast)**2))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("MAE", f"{mae:.2f}", help="Mean Absolute Error")
        with col2:
            st.metric("RMSE", f"{rmse:.2f}", help="Root Mean Square Error")
        with col3:
            r2 = 1 - (np.sum((actual - forecast)**2) / np.sum((actual - np.mean(actual))**2))
            st.metric("R²", f"{r2:.4f}", help="R-squared Score")

# ========== PAGE 6: VALUE-AT-RISK ==========
elif page == "⚠️ Value-at-Risk":
    st.markdown("<h2 style='text-align: center;'>⚠️ Value-at-Risk Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; opacity: 0.8; margin-bottom: 30px;'>Calculate potential portfolio losses and manage risk</p>", unsafe_allow_html=True)
    
    st.markdown("### 📁 Upload Risk Data")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("📌 Required columns: **date** and **price**", icon="💡")
        uploaded_var = st.file_uploader(
            "Upload CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            key='var'
        )
    
    if uploaded_var is not None:
        if uploaded_var.name.endswith('.csv'):
            df_var = pd.read_csv(uploaded_var)
        else:
            df_var = pd.read_excel(uploaded_var, engine='openpyxl')
        
        st.success("✅ Data loaded successfully!", icon="🎉")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Calculate Log Returns
        st.markdown("### 📊 Log Return Calculation")
        
        df_var = df_var.sort_values('date').reset_index(drop=True)
        df_var['LogReturn'] = np.log(df_var['price'] / df_var['price'].shift(1))
        df_var = df_var.dropna(subset=['LogReturn']).reset_index(drop=True)
        df_logreturn = df_var[['date', 'price', 'LogReturn']]
        
        with st.expander("👀 View Log Returns Data", expanded=True):
            st.dataframe(df_logreturn.head(15), use_container_width=True)
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Mean", f"{df_logreturn['LogReturn'].mean():.6f}")
        with col2:
            st.metric("📈 Std Dev", f"{df_logreturn['LogReturn'].std():.6f}")
        with col3:
            st.metric("📉 Min", f"{df_logreturn['LogReturn'].min():.6f}")
        with col4:
            st.metric("📊 Max", f"{df_logreturn['LogReturn'].max():.6f}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # VaR Calculation
        st.markdown("### 🎯 Value-at-Risk Calculator")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            value = st.number_input(
                "💵 Investment Value (IDR)",
                min_value=0,
                value=1000000,
                step=100000,
                format="%d",
                help="Total investment amount"
            )
        with col2:
            t = st.number_input(
                "📅 Time Period (days)",
                min_value=1,
                value=1,
                step=1,
                help="Investment horizon"
            )
        with col3:
            alpha = st.selectbox(
                "🎯 Confidence Level",
                [0.95, 0.99],
                format_func=lambda x: f"{int(x*100)}%",
                help="Statistical confidence level"
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔢 CALCULATE RISK", use_container_width=True):
                log_returns = df_logreturn['LogReturn']
                P_alpha = -np.percentile(log_returns, (1 - alpha) * 100)
                P_alpha_t = P_alpha * np.sqrt(t)
                loss = value * P_alpha_t
                
                # VaR Results
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 📊 Risk Assessment Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("💵 Investment", f"Rp {value:,.0f}")
                with col2:
                    st.metric("📅 Period", f"{t} day(s)")
                with col3:
                    st.metric("🎯 Confidence", f"{int(alpha*100)}%")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Loss Display
                st.markdown(f"""
                    <div style='background: linear-gradient(135deg, rgba(255, 107, 107, 0.2), rgba(255, 48, 48, 0.2)); 
                                padding: 30px; border-radius: 15px; text-align: center; 
                                border: 2px solid rgba(255, 107, 107, 0.5);'>
                        <h3 style='color: #ff6b6b; margin: 0; font-size: 1.3rem;'>⚠️ POTENTIAL LOSS (VaR)</h3>
                        <h1 style='color: white; font-size: 3rem; margin: 20px 0;'>Rp {loss:,.0f}</h1>
                        <p style='color: #e0e0e0; font-size: 1.1rem; line-height: 1.6;'>
                            There is a <strong>{int((1-alpha)*100)}%</strong> probability that losses could exceed this amount over <strong>{t} day(s)</strong>
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Risk Level
                risk_pct = (loss / value) * 100
                st.markdown("### 📌 Risk Level Interpretation")
                
                if risk_pct < 5:
                    st.success(
                        f"✅ **Low Risk:** Potential loss is {risk_pct:.2f}% of investment. "
                        f"This is acceptable for most portfolios.",
                        icon="🟢"
                    )
                elif risk_pct < 10:
                    st.info(
                        f"ℹ️ **Moderate Risk:** Potential loss is {risk_pct:.2f}% of investment. "
                        f"Consider diversification strategies.",
                        icon="🟡"
                    )
                else:
                    st.error(
                        f"⚠️ **High Risk:** Potential loss is {risk_pct:.2f}% of investment. "
                        f"Strong risk management is essential!",
                        icon="🔴"
                    )
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Log Return Distribution
                st.markdown("### 📊 Log Return Distribution with VaR")
                
                fig, ax = plt.subplots(figsize=(16, 8))
                fig.patch.set_facecolor('#0f0f19')
                ax.set_facecolor('#1a1a2e')
                
                # Histogram
                n, bins, patches = ax.hist(log_returns, bins=60, alpha=0.7, 
                                           color='#667eea', edgecolor='white', 
                                           density=True, linewidth=1.5)
                
                # Color VaR region in red
                for i, patch in enumerate(patches):
                    if bins[i] < -P_alpha:
                        patch.set_facecolor('#ff6b6b')
                        patch.set_alpha(0.8)
                
                # VaR line
                ax.axvline(-P_alpha, color='#f093fb', linestyle='--', 
                          linewidth=4, label=f'VaR {int(alpha*100)}%: {-P_alpha:.4f}',
                          alpha=0.95)
                
                # Mean line
                ax.axvline(log_returns.mean(), color='#43e97b', linestyle='--', 
                          linewidth=4, label=f'Mean: {log_returns.mean():.4f}',
                          alpha=0.95)
                
                ax.set_xlabel('Log Return', fontsize=13, color='white', fontweight='bold')
                ax.set_ylabel('Density', fontsize=13, color='white', fontweight='bold')
                ax.set_title('Log Return Distribution with VaR Threshold', 
                            fontsize=18, color='white', fontweight='bold', pad=20)
                ax.legend(fontsize=13, framealpha=0.8, facecolor='#1a1a2e', edgecolor='#667eea')
                ax.grid(True, alpha=0.2, linestyle='--', color='white')
                ax.tick_params(colors='white', labelsize=11)
                
                for spine in ax.spines.values():
                    spine.set_edgecolor('#667eea')
                    spine.set_alpha(0.5)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # VaR Metrics
                st.markdown("### 📊 VaR Metrics Summary")
                
                VaR_95 = -np.percentile(log_returns, 5)
                VaR_99 = -np.percentile(log_returns, 1)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("VaR 95% (1-day)", f"{VaR_95:.4f}", f"{VaR_95*100:.2f}%")
                with col2:
                    st.metric("VaR 99% (1-day)", f"{VaR_99:.4f}", f"{VaR_99*100:.2f}%")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Risk Management Tips
                st.markdown("### 💡 Risk Management Recommendations")
                
                tips = [
                    ("🎯", "Diversification", "Spread investments across different asset classes to reduce concentration risk"),
                    ("📊", "Position Sizing", "Limit individual position sizes based on your VaR calculations"),
                    ("⚖️", "Stop Loss", f"Consider setting stop-loss orders at {risk_pct:.1f}% below entry price"),
                    ("🔄", "Regular Monitoring", "Review VaR metrics weekly to track changing market conditions"),
                    ("📈", "Hedging Strategy", "Use derivatives or correlated assets to hedge against downside risk")
                ]
                
                for icon, title, desc in tips:
                    st.markdown(f"""
                        <div style='background: rgba(102, 126, 234, 0.1); padding: 15px; 
                                    margin: 10px 0; border-radius: 10px; border-left: 4px solid #667eea;'>
                            <span style='font-size: 1.5rem;'>{icon}</span>
                            <strong style='margin-left: 10px; font-size: 1.1rem; color: #e0e0e0;'>{title}</strong>
                            <p style='margin: 8px 0 0 45px; opacity: 0.85; line-height: 1.6; color: #c0c0d0;'>{desc}</p>
                        </div>
                    """, unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; padding: 25px; background: rgba(102, 126, 234, 0.1); 
                border-radius: 15px; border: 1px solid rgba(102, 126, 234, 0.3); margin-top: 50px;'>
        <p style='font-size: 1.2rem; margin: 0; color: #e0e0e0; letter-spacing: 2px;'>
            📈 <strong>FORRISX</strong>
        </p>
        <p style='font-size: 0.95rem; margin: 8px 0; color: #b0b0c0;'>
            Forecasting and Risk Analysis System using ARIMAX and VaR
        </p>
        <p style='opacity: 0.7; margin-top: 10px; color: #b0b0c0; font-size: 0.85rem;'>
            Advanced Financial Analytics | Created by Amanda Aulia
        </p>
    </div>
""", unsafe_allow_html=True)
