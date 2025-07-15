oords = torch.randn_like(coords)  # 使用随机噪声作为初始坐标
        # coords = coords + noise_coords  # 将噪声添加到原始坐标上
        # _, coords = self.decoder(hq, coords, edges, edge_attr)