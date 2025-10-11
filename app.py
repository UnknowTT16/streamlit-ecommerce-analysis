# ==============================================================================
# 1. 导入所有需要的库
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
from stqdm import stqdm
import os
import warnings

# (关键) 解决 KMeans 内存泄漏警告
os.environ['OMP_NUM_THREADS'] = '1'

# 分析库
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pandas.errors import SettingWithCopyWarning, DtypeWarning

# (新增) 词云图和绘图库
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# (新增) 抑制特定的Pandas警告，美化输出
warnings.filterwarnings('ignore', category=SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=DtypeWarning)

import plotly.graph_objects as go
import plotly.express as px

# ==============================================================================
# 2. 配置与辅助函数
# ==============================================================================
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8-sig')

def load_uploaded_file(uploaded_file, dtype_spec=None):
    if uploaded_file is None: return None
    file_name = uploaded_file.name
    try:
        if file_name.endswith('.parquet'): return pd.read_parquet(uploaded_file)
        elif file_name.endswith('.csv'): return pd.read_csv(uploaded_file, on_bad_lines='skip', dtype=dtype_spec)
        else: st.error(f"不支持的文件类型: {file_name}。"); return None
    except Exception as e:
        st.error(f"读取文件 '{file_name}' 时出错: {e}"); return None

@st.cache_data
def generate_wordcloud(text_series):
    if text_series.empty: return None
    full_text = " ".join(review for review in text_series.dropna())
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white", width=1200, height=600, colormap='viridis').generate(full_text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig

@st.cache_data
def perform_semantic_matching(_amazon_df, _unesco_df):
    if 'text_for_matching' not in _amazon_df.columns or not _amazon_df['text_for_matching'].notna().any():
        _amazon_df['text_for_matching'] = _amazon_df['Category'].fillna('')
    _unesco_df['text_for_matching'] = _unesco_df['Description EN'].fillna('')
    corpus = pd.concat([_amazon_df['text_for_matching'].fillna(''), _unesco_df['text_for_matching'].fillna('')], ignore_index=True)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    amazon_matrix = tfidf_matrix[:len(_amazon_df)]
    unesco_matrix = tfidf_matrix[len(_amazon_df):]
    similarity_matrix = cosine_similarity(amazon_matrix, unesco_matrix)
    return similarity_matrix, list(_unesco_df['Title EN'])

@st.cache_data
def perform_lstm_forecast(_df):
    sales_ts = _df.groupby('Date')['Amount'].sum().asfreq('D', fill_value=0)
    sales_values = sales_ts.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1)); scaled_values = scaler.fit_transform(sales_values)
    def create_dataset(data, look_back=7):
        X, y = [], [];
        for i in range(len(data) - look_back): X.append(data[i:(i + look_back), 0]); y.append(data[i + look_back, 0])
        return np.array(X), np.array(y)
    look_back = 7; X, y = create_dataset(scaled_values, look_back); X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    model = Sequential([Input(shape=(look_back, 1)), LSTM(50), Dense(1)]); model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=20, batch_size=32, verbose=0)
    last_days_scaled = scaled_values[-look_back:]; current_input = np.reshape(last_days_scaled, (1, look_back, 1)); future_predictions_scaled = []
    for _ in range(30):
        next_pred_scaled = model.predict(current_input, verbose=0); future_predictions_scaled.append(next_pred_scaled[0, 0])
        new_pred_reshaped = np.reshape(next_pred_scaled, (1, 1, 1)); current_input = np.append(current_input[:, 1:, :], new_pred_reshaped, axis=1)
    future_predictions = scaler.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1))
    last_date = sales_ts.index[-1]; future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30); fig = go.Figure()
    fig.add_trace(go.Scatter(x=sales_ts.index, y=sales_ts.values, name='历史销售额', line=dict(color='royalblue', width=2), fill='tozeroy', fillcolor='rgba(65, 105, 225, 0.2)'))
    fig.add_trace(go.Scatter(x=future_dates, y=future_predictions.flatten(), name='LSTM 预测销售额', line=dict(color='darkorange', dash='dash', width=2), fill='tozeroy', fillcolor='rgba(255, 140, 0, 0.2)'))
    fig.update_layout(title='未来30天销售额深度学习预测 (LSTM模型)', xaxis_title='日期', yaxis_title='销售额'); return fig
    
@st.cache_data
def perform_product_clustering(_df):
    required_cols = ['SKU', 'Amount', 'Qty', 'Order ID']
    if not all(col in _df.columns for col in _df.columns):
        return None, None, f"聚类分析失败：缺少必要的列。需要: {', '.join(required_cols)}"
    product_agg_df = _df.groupby('SKU').agg(total_amount=('Amount', 'sum'), total_qty=('Qty', 'sum'), order_count=('Order ID', 'nunique')).reset_index()
    features_to_cluster = ['total_amount', 'total_qty', 'order_count']
    features = product_agg_df[features_to_cluster]
    scaler = StandardScaler(); features_scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    product_agg_df['cluster'] = kmeans.fit_predict(features_scaled)
    cluster_summary = product_agg_df.groupby('cluster')[features_to_cluster].mean().sort_values(by='total_amount', ascending=False)
    hot_product_cluster_id = cluster_summary.index[0]
    hot_products = product_agg_df[product_agg_df['cluster'] == hot_product_cluster_id].sort_values(by='total_amount', ascending=False)
    return cluster_summary, hot_products, None

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

@st.cache_data
def perform_sentiment_analysis(reviews_df):
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
        return None, "错误: 未能在评论文件中找到有效的文本列。"
    reviews_df[review_column_name] = reviews_df[review_column_name].astype(str)
    reviews_df.dropna(subset=[review_column_name], inplace=True)
    reviews_df = reviews_df[reviews_df[review_column_name].str.strip() != 'None'].copy()
    stqdm.pandas(desc="正在计算情感分数") 
    reviews_df['sentiment'] = reviews_df[review_column_name].progress_apply(lambda text: analyzer.polarity_scores(text)['compound'])
    if not has_rating_col:
        reviews_df['rating'] = reviews_df['sentiment'].apply(sentiment_to_rating)
    reviews_df.rename(columns={review_column_name: 'review_text'}, inplace=True)
    return reviews_df, None

@st.cache_data
def create_category_sales_plot(_df):
    category_means = _df.groupby('Category')['Amount'].mean().sort_values(ascending=False).reset_index()
    fig = px.bar(category_means, x='Category', y='Amount', color='Category', text_auto='.2f', labels={'Category': '产品类别', 'Amount': '平均销售额 (Amount)'}, title='各产品类别平均销售额对比')
    fig.update_layout(width=800, height=500, xaxis_title_font_size=14, yaxis_title_font_size=14, title_font_size=18, template='plotly_white', showlegend=False)
    fig.update_traces(textposition='outside', textfont_size=12)
    return fig

@st.cache_resource
def get_translator(target_lang):
    return GoogleTranslator(source='en', target=target_lang)

@st.cache_data
def translate_page(_df, lang):
    df_copy = _df.copy()
    translator = get_translator(lang)
    col_name = f'Description_{lang.upper()}'
    df_copy[col_name] = _df['Description EN'].apply(lambda x: translator.translate(x) if isinstance(x, str) and x.strip() else "")
    return df_copy

# ==============================================================================
# 3. Streamlit 用户界面布局
# ==============================================================================

st.set_page_config(layout="wide")
st.title('📈 跨境选品与销售数据分析工具 ')

with st.sidebar:
    st.header("📂 上传您的数据")
    st.info("提示：为了获得更快的上传和处理速度，推荐您将大的CSV文件转换为Parquet格式。")
    uploaded_amazon = st.file_uploader('1. 上传 Amazon 销售报告', type=['csv', 'parquet'])
    uploaded_metadata = st.file_uploader('2. 上传商品元数据 (可选)', type=['csv', 'parquet'])
    uploaded_unesco = st.file_uploader('3. 上传 UNESCO 非遗数据', type=['csv', 'parquet'])
    uploaded_reviews = st.file_uploader('4. 上传 Amazon 评论数据 (可选)', type=['csv', 'parquet'])

if uploaded_amazon and uploaded_unesco:
    dtype_spec = {'ASIN': str, 'asin': str}
    amazon_df = load_uploaded_file(uploaded_amazon, dtype_spec=dtype_spec)
    unesco_df = load_uploaded_file(uploaded_unesco)
    reviews_df_loaded = load_uploaded_file(uploaded_reviews)

    if amazon_df is not None and unesco_df is not None:
        st.success("主数据文件加载成功！")
        
        with st.status("⚙️ 正在清洗和适配数据...", expanded=True) as status:
            if uploaded_metadata:
                metadata_df = load_uploaded_file(uploaded_metadata, dtype_spec=dtype_spec)
                if metadata_df is not None:
                    st.write(f"--- 发现元数据文件 '{uploaded_metadata.name}'，准备进行动态合并 ---")
                    sales_key_col, metadata_key_col = next((c for c in ['ASIN','asin'] if c in amazon_df.columns),None), next((c for c in ['asin','ASIN'] if c in metadata_df.columns),None)
                    desc_candidates = ['about_product','description','title','product_name','name','item_name']
                    metadata_desc_col = next((c for c in desc_candidates if c in metadata_df.columns), None)
                    if sales_key_col and metadata_key_col and metadata_desc_col:
                        st.write(f"✔️ 自动探测成功: 使用 '{sales_key_col}' 和 '{metadata_key_col}' 作为共同键。")
                        st.write(f"✔️ 将使用 '{metadata_desc_col}' 作为商品描述来源。")
                        amazon_df[sales_key_col] = amazon_df[sales_key_col].astype(str).str.strip()
                        metadata_df[metadata_key_col] = metadata_df[metadata_key_col].astype(str).str.strip()
                        metadata_subset = metadata_df[[metadata_key_col, metadata_desc_col]].drop_duplicates(subset=[metadata_key_col])
                        amazon_df = pd.merge(amazon_df, metadata_subset, left_on=sales_key_col, right_on=metadata_key_col, how='left')
                        total, matched = len(amazon_df), amazon_df[metadata_desc_col].notna().sum()
                        rate = (matched/total)*100 if total > 0 else 0
                        st.write(f"📊 数据合并完成！匹配成功率: **{rate:.2f}%** ({matched}/{total} 条记录)。")
                        fallback = amazon_df['Category'].fillna('')+' '+amazon_df.get('Style',pd.Series(index=amazon_df.index,dtype=str)).fillna('')
                        amazon_df['text_for_matching'] = amazon_df[metadata_desc_col].fillna(fallback)
                        st.write("--> 已为所有商品创建最终描述文本 'text_for_matching'。")
                    else:
                        st.warning("⚠️ 无法完成合并，将使用基础信息进行分析。")
                        amazon_df['text_for_matching'] = amazon_df['Category'].fillna('')
            else:
                st.write("--- 未选择元数据文件，使用基础信息进行分析 ---")
                amazon_df['text_for_matching'] = amazon_df['Category'].fillna('')+' '+amazon_df.get('Style',pd.Series(index=amazon_df.index,dtype=str)).fillna('')

            for old, new in {'Total Sales':'Amount','Product':'SKU','Quantity':'Qty','Order_ID':'Order ID'}.items():
                if old in amazon_df.columns: amazon_df.rename(columns={old:new},inplace=True)
            
            req_cols = ["Amount","Category","Date","Status","SKU","Order ID","Qty"]
            missing = [c for c in req_cols if c not in amazon_df.columns]
            if missing:
                status.update(label="数据清洗失败!", state="error", expanded=True)
                st.error(f"Amazon 文件中缺少关键列: {', '.join(missing)}")
            else:
                amazon_df.dropna(subset=["Amount", "Category", "Date"], inplace=True)
                try: amazon_df["Date"] = pd.to_datetime(amazon_df["Date"], format='%m-%d-%y')
                except ValueError: amazon_df["Date"] = pd.to_datetime(amazon_df["Date"], errors='coerce')
                amazon_df["Amount"] = pd.to_numeric(amazon_df["Amount"], errors='coerce')
                amazon_df = amazon_df[amazon_df["Status"].isin(["Shipped","Shipped - Delivered to Buyer","Completed","Pending","Cancelled"])]
                amazon_df.dropna(subset=['Date','Amount','SKU','Order ID','Qty'],inplace=True)
                all_cats = amazon_df['Category'].unique()
                non遗_products = amazon_df[amazon_df['Category'].str.contains('|'.join(all_cats), case=False, na=False)]
                status.update(label="数据清洗与适配完成!", state="complete", expanded=False)

                with st.spinner('正在进行初次数据预处理和模型计算，请稍候... (此过程仅在首次加载时运行)'):
                    cluster_summary, hot_products, cluster_error = perform_product_clustering(amazon_df)
                    keywords = ['craft','textile','embroidery','weaving','costume','dress','heritage product','handicraft']
                    relevant_unesco = unesco_df[unesco_df['Description EN'].str.contains('|'.join(keywords),case=False,na=False)]
                    
                    cosine_sim, unesco_titles = (None, None)
                    if not relevant_unesco.empty:
                        cosine_sim, unesco_titles = perform_semantic_matching(amazon_df, relevant_unesco)
                    
                    sentiment_df, sentiment_error = (None, None)
                    if reviews_df_loaded is not None:
                        sentiment_df, sentiment_error = perform_sentiment_analysis(reviews_df_loaded)
                
                st.success("核心计算完成！现在您可以快速浏览所有分析结果。")

                tabs = ["🔗 语义关联推荐", "🧠 LSTM销售预测", "🛍️ 品类表现", "🔥 热销品聚类", "💬 情感分析", "🌍 非遗描述翻译"]
                tab_sm, tab1, tab2, tab3, tab4, tab5 = st.tabs(tabs)

                with tab_sm:
                    st.header("文化关联与商品推荐")
                    if cosine_sim is not None and unesco_titles is not None:
                        st.subheader("💡 场景一: 为您的热销品寻找文化灵感")
                        if hot_products is not None and not hot_products.empty:
                            top_hot = hot_products.head(20)
                            sel_sku = st.selectbox('从您的Top 20热销商品中选择一个:', top_hot['SKU'], format_func=lambda x: f"{x} (总销售额: {top_hot.loc[top_hot['SKU']==x,'total_amount'].iloc[0]:.2f})")
                            p_indices = amazon_df.index[amazon_df['SKU']==sel_sku].tolist()
                            if p_indices:
                                p_idx = p_indices[0]
                                s_scores = sorted(list(enumerate(cosine_sim[p_idx])), key=lambda x:x[1], reverse=True)
                                top_5 = [i[0] for i in s_scores[0:5]]
                                st.write(f"与商品 **'{sel_sku}'** 最相关的5个非遗项目是:")
                                for idx in top_5:
                                    st.markdown(f"- **{unesco_titles[idx]}** (相似度: {cosine_sim[p_idx,idx]:.4f})")
                        else: st.warning("未能识别出热销商品列表。")
                        st.subheader("🚀 场景二: 根据文化元素反向寻找潜力商品")
                        sel_heritage = st.selectbox('从相关的非遗项目中选择一个:', unesco_titles)
                        if sel_heritage:
                            h_idx = unesco_titles.index(sel_heritage)
                            s_scores_h = sorted(list(enumerate(cosine_sim[:,h_idx])),key=lambda x:x[1],reverse=True)
                            top_10 = [i[0] for i in s_scores_h[:10]]
                            st.write(f"与非遗项目 **'{sel_heritage}'** 最相似的Top 10在售商品是:")
                            rec_prods = amazon_df.iloc[top_10][['SKU','Amount','Category','text_for_matching']]
                            st.dataframe(rec_prods)
                    else: st.warning("未在UNESCO文件中找到相关的非遗项目。")

                with tab1:
                    st.header("销售额深度学习预测 (LSTM)")
                    with st.spinner('正在生成LSTM预测图表...'):
                        lstm_fig = perform_lstm_forecast(amazon_df)
                    st.plotly_chart(lstm_fig, use_container_width=True)

                with tab2:
                    st.header("产品类别销售表现")
                    with st.spinner('正在生成品类表现图...'):
                        cat_fig = create_category_sales_plot(non遗_products)
                    st.plotly_chart(cat_fig)

                with tab3:
                    st.header("热销商品聚类分析")
                    if cluster_error: st.error(cluster_error)
                    elif cluster_summary is not None and hot_products is not None:
                        st.subheader("各商品簇特征均值"); st.dataframe(cluster_summary)
                        st.subheader(f"🔥 热销商品列表 (共 {len(hot_products)} 个)")
                        if len(hot_products)>20:
                            st.dataframe(hot_products.head(20))
                            if st.checkbox('显示所有热销商品',key='show_all_hot'):
                                st.dataframe(hot_products)
                        else: st.dataframe(hot_products)
                        st.download_button("下载热销商品列表 (CSV)", convert_df_to_csv(hot_products), "hot_products.csv", "text/csv")

                with tab4:
                    st.header("客户评论情感分析")
                    if uploaded_reviews:
                        if sentiment_error:
                            st.error(sentiment_error)
                        # (***关键修复点***) 使用正确的变量名 `sentiment_df`
                        elif sentiment_df is not None:
                            st.subheader("按星级筛选评论")
                            rating_range = st.slider('选择星级范围:',1,5,(4,5))
                            min_r, max_r = rating_range
                            filtered_reviews = sentiment_df[(sentiment_df['rating']>=min_r)&(sentiment_df['rating']<=max_r)]
                            st.markdown(f"**显示 {len(filtered_reviews)} 条评分为 {min_r} 到 {max_r} 星的评论**")
                            st.dataframe(filtered_reviews[['rating','review_text','sentiment']])
                            st.subheader("情感分数统计")
                            avg_filtered = filtered_reviews['sentiment'].mean() if not filtered_reviews.empty else 0
                            avg_all = sentiment_df['sentiment'].mean()
                            c1,c2 = st.columns(2)
                            c1.metric(f"所选评论 ({min_r}-{max_r} 星) 的平均情感分", f"{avg_filtered:.2f}")
                            c2.metric("所有评论的平均情感分", f"{avg_all:.2f}")
                            
                            st.subheader(f"⭐ {min_r}-{max_r} 星评论关键词词云图")
                            if not filtered_reviews.empty:
                                with st.spinner('正在根据您选择的评论生成词云图...'):
                                    wordcloud_fig = generate_wordcloud(filtered_reviews['review_text'])
                                    st.pyplot(wordcloud_fig)
                            else:
                                st.info("当前筛选范围内没有评论可用于生成词云图。")
                    else:
                        st.info("请在左侧上传评论文件 (支持 .csv 或 .parquet 格式)。")

                with tab5:
                    st.header("UNESCO 非遗项目描述多语言翻译")
                    st.markdown("将英文描述分页显示，并按需进行即时翻译。")
                    page_size, total_rows = 20, len(unesco_df)
                    total_pages = (total_rows//page_size)+(1 if total_rows%page_size>0 else 0) if total_rows>0 else 1
                    page_num = st.number_input(f'选择页码 (共 {total_pages} 页)',min_value=1,max_value=total_pages,value=1)
                    start, end = (page_num-1)*page_size, page_num*page_size
                    unesco_page = unesco_df.iloc[start:end]
                    st.markdown(f"**正在显示第 {page_num} 页, 第 {start+1} 到 {min(end,total_rows)} 条记录**")
                    display_df = unesco_page[['Title EN','Description EN']]
                    with st.expander("🌍 点击这里展开翻译选项"):
                        langs = {'中文':'zh-CN','德语':'de','法语':'fr','西班牙语':'es','日语':'ja','俄语':'ru'}
                        lang_name = st.selectbox('选择目标语言:', list(langs.keys()))
                        if lang_name:
                            lang_code = langs[lang_name]
                            if st.button(f"将当前页翻译成 {lang_name}"):
                                with st.spinner('正在翻译当前页面...'):
                                    try:
                                        trans_page = translate_page(unesco_page, lang_code)
                                        trans_col = f'Description_{lang_code.upper()}'
                                        if trans_col in trans_page.columns:
                                            display_df = trans_page[['Title EN','Description EN',trans_col]]
                                    except Exception as e: st.error(f"翻译失败: {e}")
                    st.dataframe(display_df)
else:
    st.info("👋 欢迎使用！请在左侧边栏上传 Amazon 和 UNESCO 的文件以开始分析。")