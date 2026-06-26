content = open(r"C:\Users\Administrator.SK-20241009ZRBE\Desktop\KoutuTools\auto_psd_cutout\src\grid_cutter.py", encoding="utf-8").read()

old = '''    def _detect_bg_color(self, rgb_array: np.ndarray) -> tuple[int, int, int]:
        \"\"\"从图片完整四边自动提取背景色。

        策略：
          1. 取图像最外沿 5px 宽度的四边像素
          2. 统计所有边缘像素的 RGB 直方图
          3. 取出现次数最多的颜色（mode）作为背景色
          4. 如果 mode 是纯白(255)且占比 < 80%，说明有白边框，
             则取第二高频颜色为背景色

        这比只采样四角更抗干扰 —— 即使角落有内容物，
        整条边上大部分还是背景色。
        \"\"\"
        cfg = self.cfg
        bg_color = cfg.get("bg_color", "auto")

        # 如果配置里指定了固定颜色，直接使用
        if isinstance(bg_color, (list, tuple)) and len(bg_color) == 3:
            return tuple(bg_color)

        h, w = rgb_array.shape[:2]
        border_w = 5  # 取外沿 5px

        # 从完整四边采样（不是仅角落）
        top = rgb_array[:border_w, :].reshape(-1, 3)
        bottom = rgb_array[h - border_w : h, :].reshape(-1, 3)
        left = rgb_array[border_w : h - border_w, :border_w].reshape(-1, 3)
        right = rgb_array[border_w : h - border_w, w - border_w : w].reshape(-1, 3)

        edge_pixels = np.vstack([top, bottom, left, right])

        # 量化为 16 级/通道，统计频率
        quantized = (edge_pixels // 16).astype(np.int32)
        codes, counts = np.unique(quantized, axis=0, return_counts=True)

        # 按出现次数排序
        order = np.argsort(-counts)
        codes = codes[order]
        counts = counts[order]

        # 取最高频颜色（反量化回到中心值）
        top_code = codes[0]
        bg_r = int(top_code[0] * 16 + 8)
        bg_g = int(top_code[1] * 16 + 8)
        bg_b = int(top_code[2] * 16 + 8)

        # 如果最高频是纯白(248-255)且占比 < 80%，说明有白边框
        # 取次高频颜色作为真实背景
        if bg_r >= 248 and bg_g >= 248 and bg_b >= 248 and len(codes) > 1:
            ratio = counts[0] / counts.sum()
            if ratio < 0.8:
                second = codes[1]
                bg_r = int(second[0] * 16 + 8)
                bg_g = int(second[1] * 16 + 8)
                bg_b = int(second[2] * 16 + 8)
                print(f"  检测到白边框（占比{ratio:.0f}%），取次高频色")

        bg = (min(255, bg_r), min(255, bg_g), min(255, bg_b))
        print(f"  背景色: RGB{bg}")

        self._bg_color_rgb = bg
        return bg'''

new = '''    def _detect_bg_color(self, rgb_array: np.ndarray) -> tuple[int, int, int]:
        \"\"\"从图片完整四边自动提取背景色。

        策略（多层采样，抵抗白边框干扰）：
          1. 取三层边缘：外沿 5px、向内 25px、向内 60px
          2. 每层统计颜色的出现频率
          3. 如果外层是纯白(>=248)且内层颜色不同 → 内层颜色为真实背景
          4. 否则取所有层的最高频颜色

        这能正确处理带白色边框但内背景非白的图片。
        \"\"\"
        cfg = self.cfg
        bg_color = cfg.get("bg_color", "auto")

        # 如果配置里指定了固定颜色，直接使用
        if isinstance(bg_color, (list, tuple)) and len(bg_color) == 3:
            return tuple(bg_color)

        h, w = rgb_array.shape[:2]

        # 多层采样：外(5px), 中(25px), 内(60px)
        layers = [5, 25, 60]

        def _sample_ring(offset: int) -> np.ndarray:
            \"\"\"采样距边缘 offset px 的环带像素。\"\"\"
            o = min(offset, min(h, w) // 4)
            top = rgb_array[o:o+2, o:w-o].reshape(-1, 3)
            bottom = rgb_array[h-o-2:h-o, o:w-o].reshape(-1, 3)
            left = rgb_array[o:h-o, o:o+2].reshape(-1, 3)
            right = rgb_array[o:h-o, w-o-2:w-o].reshape(-1, 3)
            return np.vstack([top, bottom, left, right])

        def _dominant_color(pixels: np.ndarray) -> tuple[int, int, int, float]:
            \"\"\"取像素集的主色（量化 16 级后最高频）。返回 (R, G, B, 占比)。\"\"\"
            quantized = (pixels // 16).astype(np.int32)
            codes, counts = np.unique(quantized, axis=0, return_counts=True)
            order = np.argsort(-counts)
            top_code = codes[order[0]]
            ratio = counts[order[0]] / counts.sum()
            r = int(top_code[0] * 16 + 8)
            g = int(top_code[1] * 16 + 8)
            b = int(top_code[2] * 16 + 8)
            return (min(255, r), min(255, g), min(255, b), ratio)

        # 逐层采样
        layer_colors = []
        for offset in layers:
            if offset < min(h, w) // 2:
                pixels = _sample_ring(offset)
                layer_colors.append(_dominant_color(pixels))

        # 选色策略：
        # 如果外层是纯白(>=248)，向内找到第一个非白的颜色
        bg = layer_colors[0][:3]
        for i in range(len(layer_colors)):
            r, g, b, ratio = layer_colors[i]
            if r >= 248 and g >= 248 and b >= 248:
                continue  # 纯白层，跳过
            if i > 0:
                prev_r, prev_g, prev_b, _ = layer_colors[i-1]
                # 如果前一层是白且这层明显不同，用这层
                if prev_r >= 248 and prev_g >= 248 and prev_b >= 248:
                    dist = ((r - prev_r)**2 + (g - prev_g)**2 + (b - prev_b)**2) ** 0.5
                    if dist > 20:
                        bg = (r, g, b)
                        print(f"  外层白色，内层背景: RGB{bg}")
                        break
            bg = (r, g, b)
            break

        print(f"  背景色: RGB{bg}")
        self._bg_color_rgb = bg
        return bg'''

content = content.replace(old, new, 1)
open(r"C:\Users\Administrator.SK-20241009ZRBE\Desktop\KoutuTools\auto_psd_cutout\src\grid_cutter.py", "w", encoding="utf-8").write(content)
print("Done")
