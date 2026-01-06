import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- 1. CORE SETUP ---
st.set_page_config(page_title='Media Intelligence', page_icon='📊', layout='wide', initial_sidebar_state='collapsed)

st.markdown('''
    <style>
    .stApp { background-color: #0b0e14; color: #e0e0e0; }
    [data-testid='stMetric'] { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; }
    .status-box { padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 5px solid #00AA13; background: #1c2128; }
    .analyst-note { font-size: 0.85rem; color: #9ca3af; font-style: italic; background: #161b22; padding: 10px; border-radius: 5px; margin-top: 5px; }
    .section-header { font-size: 1.1rem; font-weight: bold; color: #00AA13; margin-top: 20px; }
    .exec-box { background:#111827; border:1px solid #30363d; border-radius:10px; padding:20px; margin-bottom:20px; }
    </style>
''', unsafe_allow_html=True)

# --- 2. DATA ENGINE ---
@st.cache_data
def load_and_prep():
    # Load data: Date, Title, Text, Topic, Cluster, URL, Source, Reason
    df = pd.read_csv('data.csv', sep='\t')
    
    # Standardize Date
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year.astype(str)
    df['YearMonth'] = df['Date'].dt.to_period('M').astype(str)
    
    # Consolidate Clusters/Topics & Standardize
    for col in ['Cluster', 'Topic']:
        df[col] = df[col].astype(str).str.replace(r"[\'\[\]\n\r]", "", regex=True)
        # Fix the "Dual Cluster" issue by standardizing whitespace and ampersands
        df[col] = df[col].str.replace("&", "and", regex=False).str.strip()
        df[col] = df[col].str.replace(r"\s+and\s+", " and ", regex=True)

    return df


df = load_and_prep()

# --- 3. SIDEBAR (FILTERS) ---
with st.sidebar:
    st.title("Filter")
    # Using the primary Date column for the slider
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()
    
    date_range = st.date_input("Time Frame", [min_date, max_date])
    
    if len(date_range) == 2:
        selected_Clusters = st.multiselect("Topic Clusters", sorted(df['Cluster'].unique()), default=df['Cluster'].unique())
        selected_sources = st.multiselect("Sources", sorted(df['Source'].unique()), default=df['Source'].unique())

        mask = (df['Cluster'].isin(selected_Clusters)) & \
               (df['Source'].isin(selected_sources)) & \
               (df['Date'].dt.date >= date_range[0]) & \
               (df['Date'].dt.date <= date_range[1])
        f_df = df.loc[mask]
    else:
        st.stop()

# --- 4. DASHBOARD TABS ---
tab_exec, tab_matrix, tab_source, tab_pulse, tab_audit = st.tabs([
    "Executive Summary", "Intelligence Matrix", "Source Intelligence", "Narrative Pulse", "Evidence Lookup"
])

# --- TAB 1: EXECUTIVE SUMMARY ---
with tab_exec:
    st.markdown('<div class="status-box"><b>Executive Summary:</b> High-level situational awareness for leadership.</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Articles", f"{len(f_df):,}")
    c2.metric("Total Sources", f_df['Source'].nunique())
    c3.metric("Total Clusters", f_df['Cluster'].nunique())
    c4.metric("Total Topics", f_df['Topic'].nunique())

    st.markdown('''<div class="analyst-note"><b>What this shows:</b> A compressed snapshot of scale and diversity. Use this to quickly assess whether narrative space is broad (many sources/topics) or concentrated.</div>''', unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="status-box"><b>Strategic Portfolio:</b> Assessing the structural ownership of narratives.</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Narrative Hierarchy (Cluster Segmentation)</div>', unsafe_allow_html=True)

    # Get the unique clusters to split them into two rows
    unique_clusters = sorted(f_df['Cluster'].unique())
    
    # Calculate the midpoint to split the list for two rows
    midpoint = (len(unique_clusters) + 1) // 2
    row1_clusters = unique_clusters[:midpoint]
    row2_clusters = unique_clusters[midpoint:]

    # Helper function to render a row of clusters
    def render_cluster_row(clusters_to_show):
        if len(clusters_to_show) == 0:
            return
        cols = st.columns(len(clusters_to_show))
        for i, cluster in enumerate(clusters_to_show):
            with cols[i]:
                cluster_df = f_df[f_df['Cluster'] == cluster]
                st.caption(f"📍 {cluster}")
                
                fig = px.treemap(
                    cluster_df, 
                    path=['Topic'], 
                    template='plotly_dark',
                    color_discrete_sequence=[px.colors.qualitative.Pastel[i % 10]],
                    height=250 # Reduced height to allow two rows to fit on screen
                )
                fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # Render Row 1
    render_cluster_row(row1_clusters)
    
    st.write("---") # Visual separator
    
    # Render Row 2
    render_cluster_row(row2_clusters)

    st.markdown('<div class="analyst-note"><b>How to read:</b> Area size equals article count. Categories are fixed by the Defensive Hierarchy to prevent overlap.</div>', unsafe_allow_html=True)
    
    st.divider()
    
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="section-header">Cluster Share of Voice</div>', unsafe_allow_html=True)
        st.plotly_chart(px.pie(f_df, names='Cluster', hole=0.6, template='plotly_dark'), use_container_width=True)
    
    with col_r:
        st.markdown('<div class="section-header">Topic Leaderboard</div>', unsafe_allow_html=True)
        top_topics = f_df['Topic'].value_counts().nlargest(15).reset_index()
        top_topics.columns = ['Topic', 'count']
        st.plotly_chart(px.bar(top_topics, x='count', y='Topic', orientation='h', template='plotly_dark'), use_container_width=True)

# --- TAB 2: INTELLIGENCE MATRIX ---
with tab_matrix:
    st.markdown('<div class="status-box"><b>Intelligence Matrix:</b> Identifying high-momentum narratives and structural risks.</div>', unsafe_allow_html=True)
    
    mid_date = date_range[0] + (date_range[1] - date_range[0]) / 2
    p1 = f_df[f_df['Date'].dt.date < mid_date]['Topic'].value_counts()
    p2 = f_df[f_df['Date'].dt.date >= mid_date]['Topic'].value_counts()
    m_df = pd.DataFrame({'Vol': f_df['Topic'].value_counts(), 'Growth': ((p2 - p1) / p1 * 100).fillna(0)}).reset_index().rename(columns={'index': 'Topic'})
    
    st.markdown('<div class="section-header">Strategic Matrix (Volume vs. Growth)</div>', unsafe_allow_html=True)
    
    st.plotly_chart(px.scatter(m_df.nlargest(30, 'Vol'), x='Vol', y='Growth', text='Topic', size='Vol', color='Growth', color_continuous_scale='RdYlGn', template='plotly_dark', height=600), use_container_width=True)
    st.markdown('<div class="analyst-note"><b>Interpretation:</b> Top-Left (Emerging) narratives are the highest priority—low volume but high velocity. Bottom-Right (Legacy) are dominant stories losing steam.</div>', unsafe_allow_html=True)
    
    c_p1, c_p2 = st.columns(2)
    with c_p1:
        st.markdown('<div class="section-header">Narrative Persistence (Days Active)</div>', unsafe_allow_html=True)
        persist = f_df.groupby('Topic')['Date'].nunique().nlargest(10).reset_index()
        st.dataframe(persist, hide_index=True, use_container_width=True)
    with c_p2:
        st.markdown('<div class="section-header">Momentum Delta</div>', unsafe_allow_html=True)
        st.dataframe(m_df.sort_values('Growth', ascending=False).head(10), hide_index=True, use_container_width=True)

# --- TAB 3: SOURCE INTELLIGENCE ---
with tab_source:
    st.markdown('<div class="status-box"><b>Source Intelligence:</b> Analyzing narrative flow from media outlets through strategic clusters to specific topics.</div>', unsafe_allow_html=True)
    
    top_sources = f_df['Source'].value_counts().nlargest(10).index

    # --- ROW 1: THREE-TIER SANKEY DIAGRAM ---
    st.markdown('<div class="section-header">Narrative Pipeline (Source → Cluster → Topic)</div>', unsafe_allow_html=True)
    
    # 1. Prepare two-stage flow data
    s1 = f_df[f_df['Source'].isin(top_sources)].groupby(['Source', 'Cluster']).size().reset_index(name='Volume')
    s1.columns = ['source', 'target', 'value']
    
    s2 = f_df[f_df['Source'].isin(top_sources)].groupby(['Cluster', 'Topic']).size().reset_index(name='Volume')
    s2.columns = ['source', 'target', 'value']
    
    links_df = pd.concat([s1, s2])
    
    # 2. Create index mappings for all unique nodes
    all_nodes = list(pd.unique(links_df[['source', 'target']].values.ravel('K')))
    node_map = {node: i for i, node in enumerate(all_nodes)}
    
    # 3. Generate unique colors
    palette = px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24
    node_colors = [palette[i % len(palette)] for i in range(len(all_nodes))]
    
    def hex_to_rgba(hex_col, opacity=0.15):
        hex_col = hex_col.lstrip('#')
        rgb = tuple(int(hex_col[i:i+2], 16) for i in (0, 2, 4))
        return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})"

    link_colors = [hex_to_rgba(node_colors[node_map[s]], 0.15) for s in links_df['source']]

    # 4. Build Sankey
    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=all_nodes, color=node_colors),
        link=dict(source=[node_map[s] for s in links_df['source']], target=[node_map[t] for t in links_df['target']],
                  value=links_df['value'], color=link_colors)
    )])

    fig_sankey.update_layout(template='plotly_dark', height=700, margin=dict(t=30, l=10, r=10, b=30), font_size=11)
    
    
    st.plotly_chart(fig_sankey, use_container_width=True)
    
    st.markdown('''
        <div class="analyst-note">
            This diagram maps the path of news from <b>Media Source</b> (Left) to <b>Cluster</b> (Middle) to <b>Topic</b> (Right).<br>
            <b>Strategic Use:</b> Identify which sources are "narrative engines." If a specific Topic node is fed primarily by a single source path, it indicates a sort of "editorial obsession" or an exclusive investigation rather than a market-wide trend.
        </div>
    ''', unsafe_allow_html=True)

    st.divider()

    # --- ROW 2: SIDE-BY-SIDE ANALYTICS ---
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        st.markdown('<div class="section-header">Source-Cluster Concentration</div>', unsafe_allow_html=True)
        source_p_dist = f_df[f_df['Source'].isin(top_sources)].groupby(['Cluster', 'Source']).size().reset_index(name='Count')
        
        # Updated color palette: px.colors.qualitative.Bold
        st.plotly_chart(px.bar(source_p_dist, x='Cluster', y='Count', color='Source', 
                               template='plotly_dark', color_discrete_sequence=px.colors.qualitative.Bold), use_container_width=True)
        
        st.markdown('''
            <div class="analyst-note">
                This compares the total article volume of your strategic pillars across the top media outlets.<br>
                <b>Strategic Use:</b> A "fragmented" (multi-colored) bar suggests a narrative with high media consensus.
            </div>
        ''', unsafe_allow_html=True)

    with col_s2:
        st.markdown('<div class="section-header">Media Narrative Footprint (Intensity Matrix)</div>', unsafe_allow_html=True)
        
        matrix = pd.crosstab(f_df[f_df['Source'].isin(top_sources)]['Source'], f_df[f_df['Source'].isin(top_sources)]['Cluster'])
        
        # Updated color palette: 'Tealgrn' for professional intensity mapping
        fig_heat = px.imshow(
            matrix,
            labels=dict(x="Topic Cluster", y="Media Source", color="Articles"),
            color_continuous_scale='Tealgrn',
            template='plotly_dark',
            aspect="auto"
        )

        fig_heat.update_layout(margin=dict(t=10, l=0, r=0, b=0), coloraxis_showscale=False)
        
        
        st.plotly_chart(fig_heat, use_container_width=True)
        
        st.markdown('''
            <div class="analyst-note">
                This matrix ignores raw volume to show where a source chooses to concentrate its focus.<br>
                <b>Strategic Use:</b> "Hot Zones" (dark teal) represent a sort of "editorial obsessions". Use this to identify which outlets are your most consistent vocal supporters or critics in specific Topic areas.
            </div>
        ''', unsafe_allow_html=True)

# --- TAB 4: NARRATIVE PULSE ---
with tab_pulse:
    st.markdown('<div class="status-box"><b>Narrative Pulse:</b> Tracking daily topic frequency spikes and volatility.</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Narrative Pulse Map (Temporal Heatmap)</div>', unsafe_allow_html=True)
    
    pulse_data = f_df.groupby([f_df['Date'].dt.date, 'Topic']).size().reset_index(name='Volume')
    pulse_data.columns = ['Date', 'Topic', 'Volume']
    top_pulse_topics = f_df['Topic'].value_counts().nlargest(10).index
    pulse_data = pulse_data[pulse_data['Topic'].isin(top_pulse_topics)]
    
    st.plotly_chart(px.density_heatmap(pulse_data, x="Date", y="Topic", z="Volume", color_continuous_scale="Greens", template="plotly_dark", height=500), use_container_width=True)
    
    st.markdown('<div class="section-header">Volatility Index (Daily Variance)</div>', unsafe_allow_html=True)
    volat = pulse_data.groupby('Topic')['Volume'].std().reset_index().rename(columns={'Volume': 'Volatility'})
    st.plotly_chart(px.bar(volat.sort_values('Volatility'), x='Volatility', y='Topic', orientation='h', template='plotly_dark'), use_container_width=True)
    st.markdown('<div class="analyst-note"><b>Analytical Guide:</b> High volatility implies chaotic, spike-driven news (Crisis). Low volatility implies sustained messaging.</div>', unsafe_allow_html=True)

# --- TAB 5: EVIDENCE LOOKUP ---
with tab_audit:
    st.markdown('<div class="status-box"><b>Evidence Lookup:</b> Final verification layer to audit raw intelligence framing.</div>', unsafe_allow_html=True)
    
    col_t, col_s = st.columns([1, 2])
    with col_t:
        topic_filter = st.selectbox("Focus Narrative", ["All"] + sorted(f_df['Topic'].unique().tolist()))
    with col_s:
        search_q = st.text_input("Search Intelligence (Keyword in Title/Text)", "")
        
    audit_df = f_df if topic_filter == "All" else f_df[f_df['Topic'] == topic_filter]
    if search_q:
        audit_df = audit_df[audit_df['Title'].str.contains(search_q, case=False) | audit_df['Text'].str.contains(search_q, case=False)]
    
    st.dataframe(audit_df[['Date', 'Cluster', 'Topic', 'Source', 'Title', 'URL']], use_container_width=True, hide_index=True, column_config={"URL": st.column_config.LinkColumn()})
