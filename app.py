# ==============================================================================
# 1. 导入所有需要的库
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
from stqdm import stqdm
import os

os.environ['OMP_NUM_THREADS'] = '1'

from statsmodels.tsa.arima.model import ARIMA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# ==============================================================================
# 2. 配置与辅助函数 (所有函数保持不变)
# ==============================================================================
# (此处省略了所有与上一版本完全相同的辅助函数代码，以保持简洁)
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8-sig')
def perform_time_series_forecast(df):
    st.write("正在生成销售额时间序列预测...")
    sales_ts = df.groupby('Date')['Amount'].sum().asfreq('D', fill_value=0)
    model = ARIMA(sales_ts, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sales_ts.index, y=sales_ts, mode='lines', name='历史销售额', line=dict(color='royalblue', width=2), fill='tozeroy', fillcolor='rgba(65, 105, 225, 0.2)'))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast, mode='lines', name='预测销售额', line=dict(color='darkorange', dash='dash', width=2), fill='tozeroy', fillcolor='rgba(255, 140, 0, 0.2)'))
    fig.update_layout(title='未来30天销售额交互式预测 (ARIMA模型)', xaxis_title='日期', yaxis_title='销售额', legend_title='图例', template='plotly_white')
    return fig
def perform_product_clustering(df):
    st.write("正在按商品聚合数据并进行聚类分析...")
    required_cols = ['SKU', 'Amount', 'Qty', 'Order ID']
    if not all(col in df.columns for col in required_cols):
        st.error(f"聚类分析失败：缺少必要的列。需要: {', '.join(required_cols)}")
        return None, None
    product_agg_df = df.groupby('SKU').agg(total_amount=('Amount', 'sum'), total_qty=('Qty', 'sum'), order_count=('Order ID', 'nunique')).reset_index()
    features_to_cluster = ['total_amount', 'total_qty', 'order_count']
    features = product_agg_df[features_to_cluster]
    scaler = StandardScaler(); features_scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    product_agg_df['cluster'] = kmeans.fit_predict(features_scaled)
    cluster_summary = product_agg_df.groupby('cluster')[features_to_cluster].mean().sort_values(by='total_amount', ascending=False)
    hot_product_cluster_id = cluster_summary.index[0]
    hot_products = product_agg_df[product_agg_df['cluster'] == hot_product_cluster_id].sort_values(by='total_amount', ascending=False)
    return cluster_summary, hot_products
@st.cache_resource
def get_translator(target_lang):
    return GoogleTranslator(source='en', target=target_lang)
def translate_dataframe(_df, target_langs):
    st.write("正在翻译非遗项目描述...")
    df_translated = _df.copy()
    stqdm.pandas()
    for lang in target_langs:
        col_name = f'Description_{lang.upper()}'
        translator = get_translator(lang)
        def translate_cell(text):
            return translator.translate(text) if isinstance(text, str) and text.strip() else ""
        df_translated[col_name] = df_translated['Description EN'].progress_apply(translate_cell)
    return df_translated
def find_review_column(df):
    priority_cols = ['reviews.text', 'review_text', 'content', 'comment', 'review']
    for p_col in priority_cols:
        if p_col in df.columns and df[p_col].dropna().astype(str).str.strip().any(): return p_col
    possible_cols = [col for col in df.columns if any(key in str(col).lower() for key in ['text', 'review', 'content', 'comment'])]
    if possible_cols:
        string_cols = [col for col in possible_cols if df[col].dtype == 'object']
        if string_cols:
            return max(string_cols, key=lambda col: df[col].dropna().astype(str).str.len().mean())
    object_cols = df.select_dtypes(include=['object']).columns
    if not object_cols.empty:
        for col in object_cols:
            if df[col].dropna().astype(str).str.strip().any(): return col
    return None
def perform_sentiment_analysis(reviews_df):
    st.write("正在对评论数据进行情感分析...")
    def sentiment_to_rating(sentiment):
        if sentiment >= 0.5: return 5
        elif sentiment >= 0.05: return 4
        elif sentiment > -0.05: return 3
        elif sentiment > -0.5: return 2
        else: return 1
    analyzer = SentimentIntensityAnalyzer()
    has_rating_col = 'rating' in reviews_df.columns
    review_column_name = find_review_column(reviews_df)
    if review_column_name is None:
        st.error("错误: 未能在评论文件中找到有效的文本列。")
        return None
    st.info(f"自动检测到评论列为: '{review_column_name}'")
    reviews_df[review_column_name] = reviews_df[review_column_name].astype(str)
    reviews_df.dropna(subset=[review_column_name], inplace=True)
    reviews_df = reviews_df[reviews_df[review_column_name].str.strip() != 'None'].copy()
    reviews_df['sentiment'] = reviews_df[review_column_name].apply(lambda text: analyzer.polarity_scores(text)['compound'])
    if not has_rating_col:
        st.info("未在数据中找到 'rating' 列，将根据情感分数自动估算星级。")
        reviews_df['rating'] = reviews_df['sentiment'].apply(sentiment_to_rating)
    reviews_df.rename(columns={review_column_name: 'review_text'}, inplace=True)
    return reviews_df
def create_category_sales_plot(df):
    st.write("正在生成各产品类别销售对比图...")
    category_means = df.groupby('Category')['Amount'].mean().sort_values(ascending=False).reset_index()
    fig = px.bar(category_means, x='Category', y='Amount', color='Category', text_auto='.2f', labels={'Category': '产品类别', 'Amount': '平均销售额 (Amount)'}, title='各产品类别平均销售额对比')
    fig.update_layout(width=800, height=500, xaxis_title_font_size=14, yaxis_title_font_size=14, title_font_size=18, template='plotly_white', showlegend=False)
    fig.update_traces(textposition='outside', textfont_size=12)
    return fig

# ==============================================================================
# 3. Streamlit 用户界面布局
# ==============================================================================

st.set_page_config(layout="wide")
st.title('📈 跨境选品与销售数据分析工具')

with st.sidebar:
    st.header("📂 上传您的数据")
    uploaded_amazon = st.file_uploader('上传 Amazon 销售报告 (CSV)', type='csv')
    uploaded_unesco = st.file_uploader('上传 UNESCO 非遗数据 (CSV)', type='csv')
    
    # --- (关键修改) 允许上传 CSV 或 Parquet 文件 ---
    uploaded_reviews = st.file_uploader(
        '上传 Amazon 评论数据 (可选)', 
        type=['csv', 'parquet'] # 接受两种文件类型
    )

if uploaded_amazon and uploaded_unesco:
    # ... (数据加载和清洗逻辑保持不变)
    try:
        amazon_df = pd.read_csv(uploaded_amazon, on_bad_lines='skip')
        unesco_df = pd.read_csv(uploaded_unesco, on_bad_lines='skip')
    except Exception as e:
        st.error(f"文件读取失败: {e}")
    else:
        st.success("Amazon 和 UNESCO 文件上传成功！")
        
        with st.status("⚙️ 正在清洗和适配数据...", expanded=True) as status:
            if 'Total Sales' in amazon_df.columns: amazon_df.rename(columns={'Total Sales': 'Amount'}, inplace=True); st.write("✔️ 'Total Sales' -> 'Amount'")
            if 'Product' in amazon_df.columns: amazon_df.rename(columns={'Product': 'SKU'}, inplace=True); st.write("✔️ 'Product' -> 'SKU'")
            if 'Qty' not in amazon_df.columns and 'Quantity' in amazon_df.columns: amazon_df.rename(columns={'Quantity': 'Qty'}, inplace=True); st.write("✔️ 'Quantity' -> 'Qty'")
            if 'Order ID' not in amazon_df.columns and 'Order_ID' in amazon_df.columns: amazon_df.rename(columns={'Order_ID': 'Order ID'}, inplace=True); st.write("✔️ 'Order_ID' -> 'Order ID'")
            required_cols = ["Amount", "Category", "Date", "Status", "SKU", "Order ID", "Qty"]
            missing_cols = [col for col in required_cols if col not in amazon_df.columns]
            if missing_cols:
                status.update(label="数据清洗失败!", state="error", expanded=True)
                st.error(f"上传的 Amazon 文件中缺少关键列: {', '.join(missing_cols)}")
            else:
                amazon_df.dropna(subset=["Amount", "Category", "Date"], inplace=True)
                try:
                    amazon_df["Date"] = pd.to_datetime(amazon_df["Date"], format='%m-%d-%y')
                    st.write("✔️ 日期格式成功匹配: MM-DD-YY。")
                except ValueError:
                    st.write("⚠️ 日期格式不匹配 MM-DD-YY，回退到自动解析...")
                    amazon_df["Date"] = pd.to_datetime(amazon_df["Date"], errors='coerce')
                amazon_df["Amount"] = pd.to_numeric(amazon_df["Amount"], errors='coerce')
                valid_statuses = ["Shipped", "Shipped - Delivered to Buyer", "Completed", "Pending", "Cancelled"]
                amazon_df = amazon_df[amazon_df["Status"].isin(valid_statuses)]
                amazon_df.dropna(subset=['Date', 'Amount', 'SKU', 'Order ID', 'Qty'], inplace=True)
                all_categories = amazon_df['Category'].unique()
                non遗_products = amazon_df[amazon_df['Category'].str.contains('|'.join(all_categories), case=False, na=False)]
                status.update(label="数据清洗与适配完成!", state="complete", expanded=False)

                tabs = ["📊 销售预测", "🛍️ 品类表现", "🔥 热销品聚类", "💬 情感分析", "🌍 非遗描述翻译"]
                tab1, tab2, tab3, tab4, tab5 = st.tabs(tabs)

                with tab1:
                    # ...
                    st.header("销售额时间序列预测")
                    with st.spinner('正在生成预测图...'):
                        forecast_fig = perform_time_series_forecast(amazon_df)
                        st.plotly_chart(forecast_fig, use_container_width=True)
                with tab2:
                    # ...
                    st.header("产品类别销售表现")
                    with st.spinner('正在生成类别对比图...'):
                        category_fig = create_category_sales_plot(non遗_products)
                        st.plotly_chart(category_fig)
                with tab3:
                    # ...
                    st.header("热销商品聚类分析")
                    with st.spinner('正在进行聚类分析...'):
                        cluster_summary, hot_products = perform_product_clustering(amazon_df)
                        if cluster_summary is not None:
                            st.subheader("各商品簇特征均值"); st.dataframe(cluster_summary)
                            st.subheader("🔥 热销商品列表"); st.dataframe(hot_products)
                            st.download_button("下载热销商品列表 (CSV)", convert_df_to_csv(hot_products), "hot_products.csv", "text/csv")
                
                # --- (关键修改) 这是一个全新的、支持两种文件格式的 tab4 ---
                with tab4:
                    st.header("客户评论情感分析")
                    if uploaded_reviews:
                        sentiment_df = None
                        file_name = uploaded_reviews.name
                        
                        try:
                            # --- 新的智能加载逻辑 ---
                            if file_name.endswith('.parquet'):
                                # 如果是 Parquet 文件，直接读取
                                st.info(f"正在加载预处理的 Parquet 文件: '{file_name}'...")
                                sentiment_df = pd.read_parquet(uploaded_reviews)
                                st.success("Parquet 文件加载成功！")

                            elif file_name.endswith('.csv'):
                                # 如果是 CSV 文件，进行实时处理
                                st.info(f"正在实时分析上传的 CSV 文件: '{file_name}'...")
                                with st.spinner('这可能需要一些时间...'):
                                    reviews_df = pd.read_csv(uploaded_reviews)
                                    sentiment_df = perform_sentiment_analysis(reviews_df)
                                st.success("CSV 文件分析完成！")

                            # --- 后续的显示逻辑 (保持不变) ---
                            if sentiment_df is not None:
                                st.subheader("按星级筛选评论")
                                rating_range = st.slider('选择要显示的星级评分范围:', 1, 5, (4, 5))
                                min_val, max_val = rating_range
                                filtered_reviews = sentiment_df[(sentiment_df['rating'] >= min_val) & (sentiment_df['rating'] <= max_val)]
                                
                                st.markdown(f"**显示 {len(filtered_reviews)} 条评分为 {min_val} 到 {max_val} 星的评论**" if min_val != max_val else f"**显示 {len(filtered_reviews)} 条评分为 {min_val} 星的评论**")
                                st.dataframe(filtered_reviews[['rating', 'review_text', 'sentiment']])
                                
                                st.subheader("情感分数统计")
                                avg_sentiment_filtered = filtered_reviews['sentiment'].mean() if not filtered_reviews.empty else 0
                                avg_sentiment_all = sentiment_df['sentiment'].mean()
                                col1, col2 = st.columns(2)
                                metric_label = f"所选评论 ({min_val}-{max_val} 星) 的平均情感分" if min_val != max_val else f"所选评论 ({min_val} 星) 的平均情感分"
                                col1.metric(metric_label, f"{avg_sentiment_filtered:.2f}")
                                col2.metric("所有评论的平均情感分", f"{avg_sentiment_all:.2f}")

                        except Exception as e:
                            st.error(f"处理评论文件时出错: {e}")
                    else:
                        st.info("请在左侧上传一个评论文件 (支持 .csv 或 .parquet 格式) 以进行分析。")

                with tab5:
                    # ...
                    st.header("UNESCO 非遗项目描述多语言翻译")
                    st.markdown("将英文描述翻译成其他语言，以支持不同市场的卖家。")
                    available_langs = {'德语': 'de', '法语': 'fr', '西班牙语': 'es', '日语': 'ja', '俄语': 'ru'}
                    selected_langs_names = st.multiselect('选择目标语言:', list(available_langs.keys()), default=['德语', '法语'])
                    target_lang_codes = [available_langs[name] for name in selected_langs_names]
                    if st.button('开始翻译'):
                        if not target_lang_codes:
                            st.warning("请至少选择一种目标语言。")
                        else:
                            unesco_subset = unesco_df.head(20)
                            with st.spinner('翻译进行中，请稍候...'):
                                translated_df = translate_dataframe(unesco_subset, target_langs=target_lang_codes)
                                st.success("翻译完成！")
                                st.dataframe(translated_df)
                                st.download_button("下载翻译后的数据 (CSV)", convert_df_to_csv(translated_df), "unesco_translated.csv", "text/csv")
else:
    st.info("👋 欢迎使用！请在左侧边栏上传 Amazon 和 UNESCO 的 CSV 文件以开始分析。")