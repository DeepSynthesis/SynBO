import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Font, Alignment
from openpyxl.drawing.image import Image
from pathlib import Path
from openpyxl.drawing.spreadsheet_drawing import AnchorMarker, OneCellAnchor
from openpyxl.drawing.xdr import XDRPositiveSize2D
from openpyxl.utils.units import pixels_to_EMU
from openpyxl.utils import column_index_from_string, get_column_letter

from rxnopt.utils.logger import console
from rxnopt.utils.util_func import plot_SMILES, sanitize_filename


class ExcelWriter:
    def __init__(self, condition_types, opt_metrics):
        self.condition_types = condition_types
        self.opt_metrics = opt_metrics

    def write_to_excel(self, output_df, batch_id, figure_output=[], figure_path=None, save_path=None, filetype="xlsx", transpose=False):
        if filetype == "xlsx":
            wb = Workbook()
            ws = self._create_worksheet(wb, batch_id)

            self._add_data_to_worksheet(ws, output_df, transpose)

            # 2. 自动调整尺寸 (转置时调整行高，非转置时调整列宽)
            # 注意：转置后，原来的列名变成了第一列的内容
            fixed_cols = ["batch", "index", *self.opt_metrics]
            self._auto_adjust_dimensions(ws, transpose)

            if figure_output != [] and figure_path:
                console.log("exporting with specific figures...", style="green")
                for figure_type in figure_output:
                    if figure_type not in output_df.columns:
                        continue

                    col_idx_in_df = output_df.columns.get_loc(figure_type)

                    if figure_type in self.condition_types:
                        self._process_figure(
                            ws,
                            figure_type,
                            output_df,
                            figure_path,
                            col_idx_in_df,
                            transpose,
                        )
                    else:
                        console.log(
                            f"Figure output '{figure_type}' not in condition types, skipping...",
                            style="yellow",
                        )
            else:
                console.log(
                    "No figure output and path provided, exporting with names...",
                    style="green",
                )

            wb.save(save_path.with_suffix(".xlsx"))
        else:
            raise ValueError("Unknown filetype")

    def _create_worksheet(self, wb, batch_id):
        ws = wb.active
        ws.title = f"optimization in batch {batch_id}"
        return ws

    def _auto_adjust_dimensions(self, ws, transpose):
        MAX_SIZE = 70
        FONT_FACTOR = 1.3

        if not transpose:
            # --- 常规模式：调整列宽 ---
            for col_idx, col_cells in enumerate(ws.columns, 1):
                column_letter = get_column_letter(col_idx)
                max_length = 0
                for cell in col_cells:
                    try:
                        if cell.value:
                            cell_len = len(str(cell.value))
                            if cell_len > max_length:
                                max_length = cell_len
                    except:
                        pass

                adjusted_width = (max_length + 2) * FONT_FACTOR
                final_width = min(adjusted_width, MAX_SIZE)
                ws.column_dimensions[column_letter].width = final_width
        else:

            # 转置后，每一行对应原来的一列
            # ws.rows 返回的是生成器，转成 list 方便索引
            for row_idx, row_cells in enumerate(ws.rows, 1):
                max_length = 0
                for cell in row_cells:
                    try:
                        if cell.value:
                            cell_len = len(str(cell.value))
                            if cell_len > max_length:
                                max_length = cell_len
                    except:
                        pass

                # 这是一个简单的启发式计算，防止行过高
                # 这里的逻辑是：如果该行是原来 df 的列头（现在的第一列），或者是短文本，就用默认高
                # 如果是原来的固定列（如 metrics），可能需要高一点

                # 默认高度
                target_height = 25

                # 稍微根据内容长度增加一点，但不像列宽那么激进，因为行主要是横向阅读
                if max_length > 20:
                    target_height = 30

                ws.row_dimensions[row_idx].height = target_height

            # 额外处理：第一列（原来的 Header）宽度需要够宽
            max_header_len = 0
            for cell in ws["A"]:
                try:
                    if cell.value:
                        l = len(str(cell.value))
                        if l > max_header_len:
                            max_header_len = l
                except:
                    pass
            ws.column_dimensions["A"].width = (max_header_len + 2) * FONT_FACTOR

    def _add_data_to_worksheet(self, ws, output_df, transpose):
        # 基础样式配置
        # HEADER 对应 df.columns
        # DATA 对应 df 的内容

        # 如果不转置：Header 高度 35，Data 行高 25
        # 如果转置：  Header 列宽 (自动)，Data 列宽 (放图片的要宽，普通的自动)

        HEADER_HEIGHT = 35
        ROW_HEIGHT = 25

        header_font = Font(name="Arial", size=16, bold=True)
        data_font = Font(name="Arial", size=14, bold=False)
        alignment = Alignment(horizontal="center", vertical="center")

        # 获取原始数据矩阵（包含表头）
        # rows_data = [columns, row1, row2, ...]
        original_rows = list(dataframe_to_rows(output_df, index=False, header=True))

        if not transpose:
            # --- 常规写入 ---
            for i, row in enumerate(original_rows):
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
        else:
            # --- 转置写入 ---
            # original_rows 是 list of lists
            # 转置矩阵：zip(*original_rows)
            transposed_rows = list(zip(*original_rows))

            # 转置后：
            # 第1列 (Column A) 是原来的 Headers
            # 第1行 (Row 1) 是原来的 Headers 的第一个元素(通常是空或者index名，这里作为 Batch 1 的表头部分) + Batch 1 的数据

            # 在转置模式下，我们希望：
            # Column A (原来的 Header 行) 字体加粗
            # Row 1 (原来的 Column 1) 也可以加粗，或者保持普通，视需求而定。
            # 这里逻辑：原来的 Header (现在的 Col A) 用 header_font

            for i, row in enumerate(transposed_rows):
                row_idx = i + 1

                # 每一行的高度默认设置一下，后面 _auto_adjust 会微调，
                # 但如果有图片，_process_figure 会覆盖
                ws.row_dimensions[row_idx].height = ROW_HEIGHT

                for j, value in enumerate(row):
                    col_idx = j + 1
                    cell = ws.cell(row=row_idx, column=col_idx, value=value)

                    # 样式逻辑：
                    # 如果是第一列 (j==0)，对应原来的表头 -> Header Style
                    if j == 0:
                        cell.font = header_font
                        # 转置后，第一列往往需要宽一点，这里先不设，由auto_adjust处理
                    else:
                        cell.font = data_font

                    cell.alignment = alignment

    def _process_figure(self, ws, figure_type, output_df, figure_path, col_idx_in_df, transpose):
        """
        :param col_idx_in_df: 图片数据在原始 dataframe 中的列索引 (int, 0-based)
        """
        PADDING = 2  # 图片留白 (像素)

        # 尺寸定义
        # IMG_SIDE_LENGTH_PX: 图片期望显示的像素大小（正方形或长方形区域限制）
        # 我们假设图片区域大概是 100x100 像素左右的空间
        IMG_DISPLAY_SIZE = 100

        # 将像素转换为 Excel 单位的大致系数
        PX_TO_COL_WIDTH = 1 / 7.5  # 1 unit width ≈ 7.5 px
        PX_TO_ROW_HEIGHT = 0.75  # 1 point height ≈ 1.33 px -> 1 px ≈ 0.75 pt

        if not transpose:
            # ================= 非转置模式 (图片在列中) =================

            # 1. 确定位置
            # Excel列号 = df列索引 + 1 (因为Excel从1开始)
            excel_col_idx = col_idx_in_df + 1
            column_letter = get_column_letter(excel_col_idx)

            # 2. 设置单元格尺寸
            # 列宽设置 (基于字符数)
            target_col_width = IMG_DISPLAY_SIZE * PX_TO_COL_WIDTH
            current_width = ws.column_dimensions[column_letter].width
            if current_width < target_col_width:
                ws.column_dimensions[column_letter].width = target_col_width

            # 计算最终像素宽高
            cell_w_px = int(ws.column_dimensions[column_letter].width / PX_TO_COL_WIDTH)
            # 行高在下面循环中单独设置 (单位 Point)
            target_row_height_pt = IMG_DISPLAY_SIZE * PX_TO_ROW_HEIGHT
            cell_h_px = int(target_row_height_pt / PX_TO_ROW_HEIGHT)  # 回算验证

            # 3. 遍历插入
            for i in output_df.index:
                excel_row_idx = i + 2  # Header占1行，index从0开始 -> +2
                cell_address = f"{column_letter}{excel_row_idx}"

                # 设置行高
                ws.row_dimensions[excel_row_idx].height = target_row_height_pt

                # 获取文件名
                img_filename = output_df.loc[i, figure_type]

                self._insert_one_image(
                    ws,
                    figure_path,
                    figure_type,
                    img_filename,
                    cell_address,
                    excel_col_idx - 1,  # 0-based col for Anchor
                    excel_row_idx - 1,  # 0-based row for Anchor
                    cell_w_px,
                    cell_h_px,
                    PADDING,
                )

        else:
            # ================= 转置模式 (图片在行中) =================

            # 1. 确定位置
            # 原来的 df 列变成了 Excel 的行
            # Excel行号 = df列索引 + 1
            excel_row_idx = col_idx_in_df + 1

            # 2. 设置单元格尺寸
            # 行高设置 (单位 Point) - 这一行专门放图片，所以要高
            target_row_height_pt = IMG_DISPLAY_SIZE * PX_TO_ROW_HEIGHT
            ws.row_dimensions[excel_row_idx].height = target_row_height_pt

            cell_h_px = int(target_row_height_pt / PX_TO_ROW_HEIGHT)

            # 列宽设置：每一列对应原来的每一行数据
            # 需要把数据区域的列宽都撑大以放下图片
            target_col_width = IMG_DISPLAY_SIZE * PX_TO_COL_WIDTH
            cell_w_px = int(target_col_width / PX_TO_COL_WIDTH)  # 估算像素

            # 3. 遍历插入
            for i in output_df.index:
                # 原来的第 i 行数据，现在在第 i+2 列 (第1列是Header)
                excel_col_idx = i + 2
                column_letter = get_column_letter(excel_col_idx)

                # 设置列宽 (只有当这一列有图片时才需要撑大，但为了对齐通常统一设置)
                # 注意：如果同一列有多个图片类型的行，取最大值逻辑包含在内
                current_w = ws.column_dimensions[column_letter].width
                if current_w < target_col_width:
                    ws.column_dimensions[column_letter].width = target_col_width

                cell_address = f"{column_letter}{excel_row_idx}"

                # 获取文件名
                img_filename = output_df.loc[i, figure_type]

                self._insert_one_image(
                    ws,
                    figure_path,
                    figure_type,
                    img_filename,
                    cell_address,
                    excel_col_idx - 1,
                    excel_row_idx - 1,
                    cell_w_px,
                    cell_h_px,
                    PADDING,
                )

    def _insert_one_image(
        self, ws, figure_path, figure_type, img_filename, cell_address, col_idx_0, row_idx_0, cell_w_px, cell_h_px, padding
    ):
        """
        :param img_filename: 这里实际上是 DataFrame 里的值 (即 SMILES 字符串)
        """
        if pd.isna(img_filename):
            return
        # 1. 确定保存目录
        save_dir = Path(figure_path) / figure_type

        # 2. 获取安全的文件名 (模拟 plot_SMILES 内部的文件名逻辑)
        # img_filename 此时是 SMILES 字符串
        safe_name = sanitize_filename(str(img_filename))
        img_path = save_dir / f"{safe_name}.png"
        image_ready = False
        # 3. 检查逻辑：存在 -> 生成 -> 放弃
        if img_path.exists():
            # A. 图片已存在，直接使用
            image_ready = True
        else:
            # B. 图片不存在，尝试使用 SMILES 绘图
            # console.log(f"Generating image for {safe_name}...", style="blue") # 可选日志
            res = plot_SMILES(str(img_filename), str(save_dir))

            if res.get("success", False):
                # 再次确认文件确实生成了
                if img_path.exists():
                    image_ready = True
            else:
                # C. 绘图失败 (可能是无效的 SMILES，或者该列根本不是分子)
                # console.log(f"Failed to generate image for {img_filename}, keeping text.", style="yellow")
                image_ready = False
        # 4. 如果最终没有可用的图片，直接返回，保留单元格内的文本 (SMILES)
        if not image_ready:
            return
        # ================= 以下为图片插入逻辑 (仅当 image_ready=True 时执行) =================

        # 只有确定要插入图片时，才清空单元格文字
        cell = ws[cell_address]
        cell.value = ""
        try:
            img = Image(str(img_path))
            orig_w, orig_h = img.width, img.height
            # --- 计算缩放 ---
            available_w = cell_w_px - (2 * padding)
            available_h = cell_h_px - (2 * padding)
            # 防止除以零
            if orig_w == 0 or orig_h == 0:
                return
            scale_w = available_w / orig_w
            scale_h = available_h / orig_h
            scale = min(scale_w, scale_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            img.width = new_w
            img.height = new_h
            # --- 居中计算 ---
            offset_x_px = max(0, (cell_w_px - new_w) // 2)
            offset_y_px = max(0, (cell_h_px - new_h) // 2)
            # --- 设置 Anchor ---
            marker = AnchorMarker(col=col_idx_0, colOff=pixels_to_EMU(offset_x_px), row=row_idx_0, rowOff=pixels_to_EMU(offset_y_px))
            size = XDRPositiveSize2D(pixels_to_EMU(new_w), pixels_to_EMU(new_h))
            img.anchor = OneCellAnchor(_from=marker, ext=size)
            ws.add_image(img)
        except Exception as e:
            # 如果插入过程出错（例如图片文件损坏），把文字写回去以便查阅
            cell.value = img_filename
