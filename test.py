if i < len(all_countries) and j < len(all_industries):
    country = all_countries[i]
    industry = all_industries[j]
    tickers_in_group = [
        t for t in valid_tickers
        if country_df.loc[t, T] == country and map_df.loc[t, "industry_sector"] == industry
    ]
elif i == len(all_countries) and j < len(all_industries):
    # All Countries, fixed industry
    industry = all_industries[j]
    tickers_in_group = [
        t for t in valid_tickers
        if map_df.loc[t, "industry_sector"] == industry
    ]
elif i < len(all_countries) and j == len(all_industries):
    # Fixed country, All Industries
    country = all_countries[i]
    tickers_in_group = [
        t for t in valid_tickers
        if country_df.loc[t, T] == country
    ]
else:
    # All countries, all industries
    tickers_in_group = list(valid_tickers)


=======

if j == 0 and i < len(all_countries):
    ax.set_ylabel(all_countries[i], fontsize=9, rotation=0, labelpad=25, va='center')
elif j == 0 and i == len(all_countries):
    ax.set_ylabel("All Countries", fontsize=9, rotation=0, labelpad=25, va='center')

if i == 0 and j < len(all_industries):
    ax.set_title(all_industries[j], fontsize=9)
elif i == 0 and j == len(all_industries):
    ax.set_title("All Industries", fontsize=9)
