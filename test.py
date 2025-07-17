# 绘图部分开始
legend_handles = {}
for i, country in enumerate(all_countries):
    for j, industry in enumerate(all_industries):
        ax = fig.add_subplot(gs[i, j])

        if (i, j) not in cache_result:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(0, window_size - 1)
            ax.set_ylim(0, 1)
            continue

        result = cache_result[(i, j)]
        for grp, series in result.items():
            line, = ax.plot(range(window_size), series, label=grp, linewidth=1)

            # 第一次出现该组时记录 handle，用于 legend
            if grp not in legend_handles:
                legend_handles[grp] = line

        # 三条虚线标记
        ax.axvline(x=0, linestyle=":", color="black", alpha=0.4)  # T-50
        ax.axvline(x=window_size - 20, linestyle="--", color="gray", alpha=0.5)  # T-20
        ax.axvline(x=window_size - 1, linestyle=":", color="black", alpha=0.4)  # T

        ax.set_xticks([0, window_size - 20, window_size - 1])
        ax.set_xticklabels(["", "", ""])
        ax.tick_params(labelsize=6)

        if scale:
            ax.set_ylim(global_min, global_max)

        # 行名和列名
        if j == 0:
            ax.set_ylabel(country, fontsize=9, rotation=0, labelpad=25, va='center')
        if i == 0:
            ax.set_title(industry, fontsize=9)

# 添加 legend：只显示实际画出来的组
fig.legend(
    handles=[legend_handles[k] for k in legend_handles],
    labels=[k for k in legend_handles],
    loc='upper center',
    ncol=len(legend_handles)
)
fig.suptitle("Group Weighted Returns by Country and Industry", fontsize=14)
plt.show()
