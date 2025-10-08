# ==============================================================================
# 1. 导入所有需要的库
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
from stqdm import stqdm
import os

# (关键) 解决 KMeans 内存泄漏警告
os.environ['OMP_NUM_THREADS'] = '1'

# 分析库
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator

# 可视化库
import plotly.graph_objects as go
import plotly.express as px

# ==============================================================================
# 2. 配置与辅助函数
# ==============================================================================
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8-sig')

# --- (已升级) 分析函数：时间序列预测 (LSTM) ---
@st.cache_data
def perform_lstm_forecast(_df): # 使用 _df 避免与 streamlit 内部变量冲突
    st.write("正在为深度学习准备数据...")
    sales_ts = _df.groupby('Date')['Amount'].sum().asfreq('D', fill_value=0)
    sales_values = sales_ts.values.reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(sales_values)

    def create_dataset(data, look_back=7):
        X, y = [], []
        for i in range(len(data) - look_back):
            X.append(data[i:(i + look_back), 0])
            y.append(data[i + look_back, 0])
        return np.array(X), np.array(y)

    look_back = 7
    X, y = create_dataset(scaled_values, look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    st.write("正在构建并训练 LSTM 模型 (这可能需要几分钟)...")
    model = Sequential([Input(shape=(look_back, 1)), LSTM(50), Dense(1)])
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=20, batch_size=32, verbose=0)

    st.write("正在预测未来...")
    last_days_scaled = scaled_values[-look_back:]
    current_input = np.reshape(last_days_scaled, (1, look_back, 1))
    future_predictions_scaled = []
    for _ in range(30):
        next_pred_scaled = model.predict(current_input, verbose=0)
        future_predictions_scaled.append(next_pred_scaled[0, 0])
        new_pred_reshaped = np.reshape(next_pred_scaled, (1, 1, 1))
        current_input = np.append(current_input[:, 1:, :], new_pred_reshaped, axis=1)
    
    future_predictions = scaler.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1))
    
    last_date = sales_ts.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sales_ts.index, y=sales_ts.values, name='历史销售额', line=dict(color='royalblue', width=2), fill='tozeroy', fillcolor='rgba(65, 105, 225, 0.2)'))
    fig.add_trace(go.Scatter(x=future_dates, y=future_predictions.flatten(), name='LSTM 预测销售额', line=dict(color='darkorange', dash='dash', width=2), fill='tozeroy', fillcolor='rgba(255, 140, 0, 0.2)'))
    fig.update_layout(title='未来30天销售额深度学习预测 (LSTM模型)', xaxis_title='日期', yaxis_title='销售额')
    return fig

# ... (其他辅助函数与之前版本相同) ...
def perform_product_clustering(df):
    # ...
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
    # ...
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
    # ...
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
    # ...
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
    # ...
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
    uploaded_reviews = st.file_uploader('上传 Amazon 评论数据 (可选)', type=['csv', 'parquet'])

if uploaded_amazon and uploaded_unesco:
    # --- 数据加载与健壮性清洗 ---
    try:
        # 在这里，我们不指定dtype，让pandas自动推断，后续再处理
        amazon_df = pd.read_csv(uploaded_amazon, on_bad_lines='skip')
        unesco_df = pd.read_csv(uploaded_unesco, on_bad_lines='skip')
    except Exception as e:
        st.error(f"文件读取失败: {e}")
    else:
        st.success("Amazon 和 UNESCO 文件上传成功！")
        
        with st.status("⚙️ 正在清洗和适配数据...", expanded=True) as status:
            # ... (智能重命名逻辑保持不变)
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
                
                # 消除日期格式警告
                try:
                    amazon_df["Date"] = pd.to_datetime(amazon_df["Date"], format='%m-%d-%y')
                    st.write("✔️ 日期格式成功匹配: MM-DD-YY。")
                except ValueError:
                    st.write("⚠️ 日期格式不匹配 MM-DD-YY，回退到自动解析...")
                    amazon_df["Date"] = pd.to_datetime(amazon_df["Date"], errors='coerce')
                
                # ... (后续清洗逻辑保持不变)
                amazon_df["Amount"] = pd.to_numeric(amazon_df["Amount"], errors='coerce')
                valid_statuses = ["Shipped", "Shipped - Delivered to Buyer", "Completed", "Pending", "Cancelled"]
                amazon_df = amazon_df[amazon_df["Status"].isin(valid_statuses)]
                amazon_df.dropna(subset=['Date', 'Amount', 'SKU', 'Order ID', 'Qty'], inplace=True)
                all_categories = amazon_df['Category'].unique()
                non遗_products = amazon_df[amazon_df['Category'].str.contains('|'.join(all_categories), case=False, na=False)]
                status.update(label="数据清洗与适配完成!", state="complete", expanded=False)

                # --- 创建选项卡 ---
                tabs = ["🧠 LSTM销售预测", "🛍️ 品类表现", "🔥 热销品聚类", "💬 情感分析", "🌍 非遗描述翻译"]
                tab1, tab2, tab3, tab4, tab5 = st.tabs(tabs)

                # --- (关键修改) 将原tab1的ARIMA预测替换为LSTM预测 ---
                with tab1:
                    st.header("销售额深度学习预测 (LSTM)")
                    st.markdown("使用 LSTM 深度学习模型，根据历史销售数据预测未来30天的销售趋势。")
                    # 使用缓存来避免每次切换Tab都重新训练模型
                    lstm_forecast_fig = perform_lstm_forecast(amazon_df)
                    st.plotly_chart(lstm_forecast_fig, use_container_width=True)

                # --- (其他Tab保持不变) ---
                with tab2:
                    # ...
                    st.header("产品类别销售表现")
                    st.markdown("对比不同产品类别的平均销售额。")
                    with st.spinner('正在生成类别对比图...'):
                        category_fig = create_category_sales_plot(non遗_products)
                        st.plotly_chart(category_fig)

                with tab3:
                    # ...
                    st.header("热销商品聚类分析")
                    st.markdown("通过K-Means聚类，根据总销售额、总销量和订单数找出热门商品。")
                    with st.spinner('正在进行聚类分析...'):
                        cluster_summary, hot_products = perform_product_clustering(amazon_df)
                        if cluster_summary is not None:
                            st.subheader("各商品簇特征均值"); st.dataframe(cluster_summary)
                            st.subheader("🔥 热销商品列表"); st.dataframe(hot_products)
                            st.download_button("下载热销商品列表 (CSV)", convert_df_to_csv(hot_products), "hot_products.csv", "text/csv")
                
                with tab4:
                    # ...
                    st.header("客户评论情感分析")
                    if uploaded_reviews:
                        # ...
                        sentiment_df = None
                        file_name = uploaded_reviews.name
                        try:
                            if file_name.endswith('.parquet'):
                                st.info(f"正在加载预处理的 Parquet 文件: '{file_name}'...")
                                sentiment_df = pd.read_parquet(uploaded_reviews)
                                st.success("Parquet 文件加载成功！")
                            elif file_name.endswith('.csv'):
                                st.info(f"正在实时分析上传的 CSV 文件: '{file_name}'...")
                                with st.spinner('这可能需要一些时间...'):
                                    reviews_df = pd.read_csv(uploaded_reviews)
                                    sentiment_df = perform_sentiment_analysis(reviews_df)
                                st.success("CSV 文件分析完成！")
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
                    available_langs = {'德语': 'de', '法语': 'fr', '西班牙语': 'es', '日语': 'ja', '俄语': 'ru', '中文': 'zh-cn'}
                    selected_langs_names = st.multiselect('选择目标语言:', list(available_langs.keys()), default=['德语', '法语', '中文'])
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