from Visualization import MultiRecorderPlot, RecorderPlot

if __name__ == "__main__":
    f1 = "datas/ET-C1-50-alpha=5-beta=5.json"
    f2 = "datas/ET-C1-50-alpha=5-beta=10.json"
    f3 = "datas/ET-C1-50-alpha=5-beta=15.json"
    f4 = "datas/ET-C1-50-alpha=10-beta=5.json"
    f5 = "datas/ET-C1-50-alpha=10-beta=10.json"
    f6 = "datas/ET-C1-50-alpha=10-beta=15.json"
    f7 = "datas/ET-C1-50-alpha=15-beta=5.json"
    f8 = "datas/ET-C1-50-alpha=15-beta=10.json"
    f9 = "datas/ET-C1-50-alpha=15-beta=15.json"
    fpd = "mento_carlo_data/PD-C1-50-rate=17.json"

    rp = RecorderPlot(f5)
    rp.draw_RMSE()
    # mrp = MultiRecorderPlot(f1, f2, f3, f4, f5, f6, f7, f8, f9)
    # mrp = MultiRecorderPlot(f2, f5, f8, f4, f6, fpd)
    # mrp.draw_multi_RMSE(mrp.follower_nodes)
    # mrp.trans_rate_plot(mode="bar", params=[(5, 10), (10, 10), (15, 10), (10, 5), (10, 10), (10, 15), (None, None)])

    # files = []
    # for alpha in range(5, 16):
    #     files.append(f"mento_carlo_data/ET-C1-8-alpha={alpha}-beta=15.json")
    #
    # mrp_r = MultiRecorderPlot(*files)
    # mrp_r.trans_rate_plot(mode="plot", info="alpha")
