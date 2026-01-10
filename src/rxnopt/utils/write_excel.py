import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Font, Alignment
from openpyxl.drawing.image import Image
from pathlib import Path
from openpyxl.drawing.spreadsheet_drawing import AnchorMarker, OneCellAnchor
from openpyxl.drawing.xdr import XDRPositiveSize2D
from openpyxl.utils.units import pixels_to_EMU
from openpyxl.utils import column_index_from_string

from rxnopt.utils.logger import console


class ExcelWriter:
    def __init__(self, condition_types, opt_metrics):
        self.condition_types = condition_types
        self.opt_metrics = opt_metrics

    def write_to_excel(self, output_df, batch_id, figure_output=[], figure_path=None, save_path=None, filetype="xlsx"):
        if filetype == "xlsx":
            wb = Workbook()
            ws = self._create_worksheet(wb, batch_id)

            # 1. 填充数据并应用基础样式
            self._add_data_to_worksheet(ws, output_df)

            # 2. 自动调整列宽
            fixed_length_col = [
                chr(ord("A") + output_df.columns.get_loc(i)) for i in ["batch", "index", *self.opt_metrics] if i in output_df.columns
            ]
            self._auto_adjust_columns(ws, fixed_length_col)

            # 3. 处理图片插入
            if figure_output != [] and figure_path:
                console.log("exporting with specific figures...", style="green")
                for figure_type in figure_output:
                    if figure_type not in output_df.columns:
                        continue

                    column_idx_letter = chr(ord("A") + output_df.columns.get_loc(figure_type))
                    if figure_type in self.condition_types:
                        self._process_figure(ws, figure_type, output_df, figure_path, column_idx_letter)
                    else:
                        console.log(f"Figure output '{figure_type}' not in condition types, skipping...", style="yellow")
            else:
                console.log("No figure output and path provided, exporting with names...", style="green")

            wb.save(save_path.with_suffix(".xlsx"))
        else:
            raise ValueError("Unknown filetype")

    def _create_worksheet(self, wb, batch_id):
        ws = wb.active
        ws.title = f"optimization in batch {batch_id}"
        return ws

    def _auto_adjust_columns(self, ws):
        MAX_WIDTH_UNITS = 70
        FONT_FACTOR = 1.3

        for col in ws.columns:
            column_letter = col[0].column_letter
            max_length = 0

            for cell in col:
                try:
                    if cell.value:
                        cell_len = len(str(cell.value))
                        if cell_len > max_length:
                            max_length = cell_len
                except:
                    pass

            adjusted_width = (max_length + 2) * FONT_FACTOR
            final_width = min(adjusted_width, MAX_WIDTH_UNITS)
            ws.column_dimensions[column_letter].width = final_width

    def _add_data_to_worksheet(self, ws, output_df):
        HEADER_HEIGHT = 35
        ROW_HEIGHT = 25
        header_font = Font(name="Arial", size=16, bold=True)
        data_font = Font(name="Arial", size=14, bold=False)
        alignment = Alignment(horizontal="center", vertical="center")

        rows = list(dataframe_to_rows(output_df, index=False, header=True))

        for i, row in enumerate(rows):
            row_idx = i + 1
            if i == 0:
                ws.row_dimensions[row_idx].height = HEADER_HEIGHT
                current_font = header_font
            else:
                ws.row_dimensions[row_idx].height = ROW_HEIGHT
                current_font = data_font

            for j, value in enumerate(row):
                cell = ws.cell(row=row_idx, column=j + 1, value=value)
                cell.font = current_font
                cell.alignment = alignment

    def _process_figure(self, ws, figure_type, output_df, figure_path, column_idx_letter):
        # 预设参数
        TARGET_ROW_HEIGHT_PT = 80  # 设定放图片的行高 (单位: Point)
        TARGET_COL_WIDTH_UNITS = 30  # 设定放图片的列宽初始值 (单位: Char width)
        PADDING = 2  # 图片距离单元格边框的留白 (像素)

        # 1. 先设置该列的列宽，确保有足够的空间
        # 注意：如果该列原本的文字很长导致列宽很大，保留较大的那个
        current_width = ws.column_dimensions[column_idx_letter].width
        if current_width < TARGET_COL_WIDTH_UNITS:
            ws.column_dimensions[column_idx_letter].width = TARGET_COL_WIDTH_UNITS

        # 重新获取最终列宽 (Excel单位) 并转换为像素
        # Excel列宽单位到像素的大致转换: (width * 7) 是一个近似值，更精确通常是 width * 7.5 或 8 取决于字体
        # 这里使用 openpyxl 内部逻辑的逆运算估算像素：Approx 7px per unit for standard font
        # 为了保险起见，我们手动定义一个转换系数。通常 Arial 10pt 下 1 unit ≈ 7-8 pixels.
        # 你的字体较大 (14pt)，但列宽单位是基于默认字体的字符数的。
        PIXEL_PER_COL_UNIT = 7.5
        final_col_width_units = ws.column_dimensions[column_idx_letter].width
        cell_w_px = int(final_col_width_units * PIXEL_PER_COL_UNIT)

        # 行高单位 (Points) 到像素 (Pixels) 的转换: 1 Point = 1.333 Pixels (96 DPI)
        cell_h_px = int(TARGET_ROW_HEIGHT_PT * 1.3333)

        for i in output_df.index:
            excel_row_idx = i + 2
            cell_address = f"{column_idx_letter}{excel_row_idx}"

            img_filename = output_df.loc[i, figure_type]
            if pd.isna(img_filename):
                continue

            img_path = Path(figure_path) / Path(f"{figure_type}/{img_filename}.png")

            if not img_path.exists():
                # 尝试 jpg 扩展名或者不带扩展名的情况
                if not img_path.exists():
                    console.log(f"{img_path} does not exist, skipping...", style="yellow")
                    continue

            # 清空单元格文字内容
            cell = ws[cell_address]
            cell.value = None

            # 设置该行行高
            ws.row_dimensions[excel_row_idx].height = TARGET_ROW_HEIGHT_PT

            try:
                img = Image(str(img_path))
                orig_w, orig_h = img.width, img.height

                # --- 计算缩放比例 ---
                # 目标是：图片放入 (cell_w_px, cell_h_px) 的框内，且保留 PADDING
                available_w = cell_w_px - (2 * PADDING)
                available_h = cell_h_px - (2 * PADDING)

                # 计算宽和高的缩放比
                scale_w = available_w / orig_w
                scale_h = available_h / orig_h

                # 取较小的缩放比，保证图片完整显示且顶满一边 (Contain 模式)
                scale = min(scale_w, scale_h)

                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)

                img.width = new_w
                img.height = new_h

                # --- 计算居中偏移量 ---
                # OneCellAnchor 的偏移量是相对于单元格左上角的
                col_idx = column_index_from_string(column_idx_letter) - 1
                row_idx = excel_row_idx - 1

                offset_x_px = (cell_w_px - new_w) // 2
                offset_y_px = (cell_h_px - new_h) // 2

                # 确保偏移量不为负
                offset_x_px = max(0, offset_x_px)
                offset_y_px = max(0, offset_y_px)

                # --- 设置锚点 ---
                marker = AnchorMarker(col=col_idx, colOff=pixels_to_EMU(offset_x_px), row=row_idx, rowOff=pixels_to_EMU(offset_y_px))
                size = XDRPositiveSize2D(pixels_to_EMU(new_w), pixels_to_EMU(new_h))

                img.anchor = OneCellAnchor(_from=marker, ext=size)
                ws.add_image(img)

            except Exception as e:
                console.log(f"Error adding image {img_filename}: {e}", style="red")
