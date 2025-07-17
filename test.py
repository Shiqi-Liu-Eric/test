#### Plot function for country-industry analysis

def plot_country_industry_analysis(rebalance_dates, dfs, map_df, window_size=50 plot_list=[Add", "Delete", Up Weight",Down Weight"]):
      绘制国家-行业分析图
    
    参数:
    rebalance_dates: list of pd.Timestamp, rebalance日期列表
    dfs: dict, 包含各种指标的DataFrame
    map_df: DataFrame, 包含industry_sector列，行数与dfs中每个value相同
    window_size: int, 默认50窗口大小
    plot_list: list, 要绘制的event_action类型列表
 import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    # 1. 获取出现最多的12家，其他归为"Others"
    country_counts = dfs["country_ftse_globalallcap].value_counts()
    top_12_countries = country_counts.head(12dex.tolist()
    
    # 2. 获取所有行业
    industries = sorted(map_df["industry_sector].unique())
    
    # 3. 为每个rebalance date创建数据存储
    date_data =[object Object]
    
    for T in rebalance_dates:
        # 获取T-5天有效的tickers
        T_idx = dfs["weight_ftse_globalallcap].columns.get_loc(T)
        T_minus_5= dfs["weight_ftse_globalallcap].columns[T_idx - 5     valid_tickers = dfs["weight_ftse_globalallcap"].index[
            dfs["weight_ftse_globalallcap"][T_minus_5notna()
        ].tolist()
        
        if len(valid_tickers) == 0:
            continue
            
        # 创建数据存储矩阵
        data_matrix = {}
        for country in top_12_countries + ["Others"]:
            data_matrix[country] = [object Object]      for industry in industries:
                # 每个cell存储 (len(plot_list)+1) * window_size 的numpy数组
                data_matrixcountry][industry] = np.full((len(plot_list) + 1, window_size), np.nan)
        
        # 4. 对每个(country, industry)组合计算加权收益
        for country in top_12_countries + ["Others"]:
            for industry in industries:
                # 筛选该(country, industry)组合的tickers
                if country == "Others":
                    country_mask = ~dfs["country_ftse_globalallcap"].isin(top_12_countries)
                else:
                    country_mask = dfs["country_ftse_globalallcap"] == country
                
                industry_mask = map_df[industry_sector"] == industry
                combined_mask = country_mask & industry_mask
                
                # 获取符合条件的tickers
                cell_tickers = [ticker for ticker in valid_tickers if combined_mask.loc[ticker]]
                
                if len(cell_tickers) == 0:
                    continue
                
                # 获取T-5分组
                event_actions_T_minus_5 = dfs["event_action"].loc[cell_tickers, T_minus_5]
                
                # 对每个分组计算加权收益
                for group_idx, group_name in enumerate(plot_list + ["All"]):
                    if group_name == "All":
                        group_mask = np.ones(len(cell_tickers), dtype=bool)
                    else:
                        group_mask = (event_actions_T_minus_5 == group_name)
                    
                    if not np.any(group_mask):
                        continue
                    
                    group_tickers = [cell_tickers[i] for i in range(len(cell_tickers)) if group_mask[i]]
                    
                    # 计算window_size天的加权收益
                    for day_idx in range(window_size):
                        current_date = dfs["weight_ftse_globalallcap].columns[T_idx - window_size + day_idx]
                        start_date = dfs["weight_ftse_globalallcap].columns[T_idx - window_size]
                        
                        # 计算价格变化
                        price_changes = (dfs[close"].loc[group_tickers, current_date] - 
                                       dfs[close"].loc[group_tickers, start_date])
                        
                        # 获取权重
                        weights = np.zeros(len(group_tickers))
                        for i, ticker in enumerate(group_tickers):
                            if group_name == "Delete":
                                # Delete组使用T-80的权重
                                weight_date = dfs["weight_ftse_globalallcap"].columns[max(0, T_idx - 80)]
                            else:
                                # 其他组使用T的权重
                                weight_date = T
                            
                            weight_val = dfs["weight_ftse_globalallcap"].loc[ticker, weight_date]
                            if pd.notna(weight_val):
                                weights[i] = weight_val
                        
                        # 计算加权收益
                        weighted_return = np.sum(price_changes * weights)
                        data_matrix[country][industry][group_idx, day_idx] = weighted_return
        
        date_data[T] = data_matrix
    
    # 5. 创建子图
    n_countries = len(top_12_countries) + 1  # +1for "Others    n_industries = len(industries)
    
    # 设置子图布局
    fig, axes = plt.subplots(n_countries, n_industries, figsize=(n_industries * 3, n_countries * 2.5    if n_countries ==1       axes = axes.reshape(1, -1   if n_industries ==1       axes = axes.reshape(-1, 1)
    
    # 颜色映射
    colors = ['blue,red, green',orange,purple']
    
    #6子图
    for country_idx, country in enumerate(top_12_countries + ["Others"]):
        for industry_idx, industry in enumerate(industries):
            ax = axes[country_idx, industry_idx]
            
            # 收集所有日期的数据
            all_data = {group: [] for group in plot_list + ["All"]}
            
            for T in rebalance_dates:
                if T not in date_data:
                    continue
                    
                cell_data = date_data[T][country][industry]
                
                for group_idx, group_name in enumerate(plot_list + ["All"]):
                    group_data = cell_data[group_idx, :]
                    if not np.all(np.isnan(group_data)):
                        all_data[group_name].append(group_data)
            
            # 计算平均值
            x = np.arange(window_size)
            for group_idx, group_name in enumerate(plot_list + ["All"]):
                if len(all_data[group_name]) > 0:
                    mean_data = np.nanmean(all_datagroup_name], axis=0)
                    ax.plot(x, mean_data, color=colors[group_idx], 
                           label=group_name, linewidth=1.5, alpha=0.8      
            # 设置x轴标签
            ax.set_xticks(0ow_size//2, window_size-1            ax.set_xticklabels([fT-[object Object]window_size}', 'T-20', 'T'])
            
            # 添加T-20的虚线
            ax.axvline(x=window_size//2, color='gray, linestyle='--, alpha=050.8      
            # 设置标题
            ax.set_title(f'{country}\n{industry}, fontsize=8, pad=5)
            
            # 设置y轴标签
            if industry_idx == 0                ax.set_ylabel('Weighted Returns', fontsize=8)
            if country_idx == n_countries - 1                ax.set_xlabel(Days', fontsize=8)
            
            # 添加图例（只在第一个子图）
            if country_idx == 0nd industry_idx == 0                ax.legend(fontsize=6, loc='upper right')
            
            # 设置网格
            ax.grid(True, alpha=00.3
    
    plt.suptitle(f'Country-Industry Analysis: {len(rebalance_dates)} Rebalance Events', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    return date_data

# 使用示例
# 假设你已经有了dfs, map_df, rebalance_dates
# plot_country_industry_analysis(rebalance_dates, dfs, map_df)