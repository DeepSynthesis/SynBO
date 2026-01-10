from openpyxl import Workbook
from pathlib import Path
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Font, Alignment
from openpyxl.drawing.image import Image
from openpyxl.worksheet.table import Table, TableStyleInfo
import pandas as pd
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

            # 2. 自动调整列宽
            fixed_length_col = [
                chr(ord("A") + output_df.columns.get_loc(i)) for i in ["batch", "index", *self.opt_metrics] if i in output_df.columns
            ]
            self._auto_adjust_columns(ws, fixed_length_col)

            # 3. 移除原来的 _apply_table_style，不再生成Table对象

            # 4. 处理图片插入
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
        # 适当调整列宽倍率，因为字体变大了
        max_width = 0
        for col in ws.columns:
            # 简单估算字符长度
            current_max = max((len(str(cell.value)) for cell in col if cell.value), default=0)
            max_width = current_max if current_max > max_width else max_width

        adjusted_width = (max_width + 2) * 1.3  # 稍微增加一点倍率以适应大字体
        for col in ws.columns:
            column_letter = col[0].column_letter
            ws.column_dimensions[column_letter].width = adjusted_width

        for n_col in number_col:
            ws.column_dimensions[n_col].width = 20  # 稍微加宽一点

    def _add_data_to_worksheet(self, ws, output_df):
        # 定义样式常量
        # OpenPyXL height implies points. 30px approx 22.5pt, 25px approx 18.75pt
        HEADER_HEIGHT = 22.5
        ROW_HEIGHT = 18.75

        header_font = Font(name="Arial", size=16, bold=True)
        data_font = Font(name="Arial", size=14, bold=False)

        # 统一居中
        alignment = Alignment(horizontal="center", vertical="center")

        # 使用 dataframe_to_rows 获取所有行（包含表头）
        rows = list(dataframe_to_rows(output_df, index=False, header=True))

        for i, row in enumerate(rows):
            row_idx = i + 1  # OpenPyXL 是 1-based 索引

            # 设置行高和字体选择
            if i == 0:
                # 表头行
                ws.row_dimensions[row_idx].height = HEADER_HEIGHT
                current_font = header_font
            else:
                # 数据行
                ws.row_dimensions[row_idx].height = ROW_HEIGHT
                current_font = data_font

            for j, value in enumerate(row):
                cell = ws.cell(row=row_idx, column=j + 1, value=value)
                cell.font = current_font
                cell.alignment = alignment

    def _process_figure(self, ws, figure_type, output_df, figure_path, column_idx_letter):
        # 移除之前硬编码修改第一行高度的代码: ws.row_dimensions[1].height = 50

        for i in output_df.index:
            # i 是 DataFrame 的索引，但在 Excel 中，数据从第2行开始（第1行是表头）
            # 所以 Excel 行号是 i + 2
            excel_row_idx = i + 2

            img_filename = output_df.loc[i, figure_type]
            if pd.isna(img_filename):
                continue

            img_path = Path(figure_path) / Path(f"{figure_type}/{img_filename}.png")

            if not img_path.exists():
                console.log(f"{img_path} does not exist, skipping...", style="yellow")
                continue

            # 清空单元格文字值，准备放图片
            ws[f"{column_idx_letter}{excel_row_idx}"].value = None

            try:
                img = Image(str(img_path))  # 确保路径是字符串

                # 调整图片大小 (根据原逻辑保持不变)
                img.width = int(img.width / 5)
                img.height = int(img.height / 5)

                ws.add_image(img, f"{column_idx_letter}{excel_row_idx}")

                # 只有当图片高度大于当前的默认行高（25px/18.75pt）时，才撑大行高
                # 注意：img.height 单位通常是像素，row_dimensions 是磅。
                # 这是一个简单的转换估算，或者直接比较数值（OpenPyXL有时处理单位比较模糊）
                current_height = ws.row_dimensions[excel_row_idx].height
                if current_height is None:
                    current_height = 18.75

                # 将图片像素高度转换为粗略的磅数 (px * 0.75) 进行比较，或者直接保留原逻辑
                # 原逻辑: max(img.height * 0.8, ...)
                ws.row_dimensions[excel_row_idx].height = max(img.height * 0.8, current_height)

                ws.column_dimensions[column_idx_letter].width = img.width * 0.2
            except Exception as e:
                console.log(f"Failed to add image for {figure_type} at row {excel_row_idx}: {str(e)}", style="red")
