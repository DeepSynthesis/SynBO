import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Font, Alignment
from openpyxl.drawing.image import Image
from pathlib import Path

from rxnopt.utils.logger import console


class ExcelWriter:
    def __init__(self, condition_types, opt_metrics):
        self.condition_types = condition_types
        self.opt_metrics = opt_metrics

    def write_to_excel(self, output_df, batch_id, figure_output=[], figure_path=None, save_path=None, filetype="xlsx"):
        if filetype == "xlsx":
            wb = Workbook()
            ws = self._create_worksheet(wb, batch_id)

            # 1. 填充数据并应用基础样式（字体、字号、行高）
            self._add_data_to_worksheet(ws, output_df)

            # 2. 自动调整列宽 (限制最大宽度为 500px)
            # 这里的 fixed_length_col 逻辑保留，但在新的 auto_adjust 中作为参考或特定处理
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

    def _auto_adjust_columns(self, ws, number_col):
        """
        自动调整列宽，确保内容显示完全，但设置最大上限。
        OpenPyXL width unit ≈ 1 character width.
        Approx conversion: 1 unit width ≈ 7 pixels (Standard font size 11).
        Since we use larger font (size 14/16), the factor changes, but we limit logically.
        Let's assume max limit 500px ≈ 60-70 width units.
        """
        MAX_WIDTH_UNITS = 70
        FONT_FACTOR = 1.3  # 因为用了14号/16号字体，比默认字体宽，需要乘个系数

        for col in ws.columns:
            column_letter = col[0].column_letter
            max_length = 0

            # 遍历该列所有单元格寻找最长内容
            for cell in col:
                try:
                    if cell.value:
                        cell_len = len(str(cell.value))

                        if cell_len > max_length:
                            max_length = cell_len
                except:
                    print("wawawa")

            adjusted_width = (max_length + 2) * FONT_FACTOR

            final_width = min(adjusted_width, MAX_WIDTH_UNITS)
            ws.column_dimensions[column_letter].width = final_width

    def _add_data_to_worksheet(self, ws, output_df):
        HEADER_HEIGHT = 35
        ROW_HEIGHT = 25

        header_font = Font(name="Arial", size=16, bold=True)
        data_font = Font(name="Arial", size=14, bold=False)

        alignment = Alignment(horizontal="left", vertical="center")

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
        for i in output_df.index:
            excel_row_idx = i + 2

            img_filename = output_df.loc[i, figure_type]
            if pd.isna(img_filename):
                continue

            img_path = Path(figure_path) / Path(f"{figure_type}/{img_filename}.png")

            if not img_path.exists():
                console.log(f"{img_path} does not exist, skipping...", style="yellow")
                continue

            ws[f"{column_idx_letter}{excel_row_idx}"].value = None

            try:
                img = Image(str(img_path))

                # 调整图片显示尺寸
                img.width = int(img.width / 5)
                img.height = int(img.height / 5)

                ws.add_image(img, f"{column_idx_letter}{excel_row_idx}")

                current_height = ws.row_dimensions[excel_row_idx].height
                if current_height is None:
                    current_height = 18.75

                ws.row_dimensions[excel_row_idx].height = max(img.height * 0.8, current_height)

                # 图片列宽处理：
                # 如果图片宽度所需的列宽 超过了 500px限制，这里是否要为了图片打破限制？
                # 如果想保持500px限制，就不动 width；如果想让图片完整显示，可以取消下面的 min 限制。
                # 下面逻辑是：让列宽适应图片，但也限制最大值（为了保持表格整齐）
                img_col_width = img.width * 0.14  # 粗略换算 px to column width unit
                current_col_width = ws.column_dimensions[column_idx_letter].width

                # 取 图片所需宽度 和 500px限制(约70) 中的较小值，同时不能比文字宽度小
                limit_width = 70
                new_width = min(max(img_col_width, current_col_width), limit_width)

                ws.column_dimensions[column_idx_letter].width = new_width

            except Exception as e:
                console.log(f"Failed to add image for {figure_type} at row {excel_row_idx}: {str(e)}", style="red")
