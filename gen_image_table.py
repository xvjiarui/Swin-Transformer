import argparse
import math
import os
import os.path as osp
from pathlib import Path

from PIL import Image
from functools import cmp_to_key

class Element:
    """A data element of a row in a table."""

    def __init__(self, htmlCode=''):
        self.htmlCode = htmlCode
        self.isHeader = False
        self.drawBorderColor = ''

    def imgToHTML(self, img_path, width=100, overlay_path=None):
        res = '<img src="' + img_path.strip().lstrip() + '" width="' + str(
            width) + 'px" '
        if self.drawBorderColor:
            res += 'style="border: 10px solid ' + self.drawBorderColor + '" '
        if overlay_path:
            res += 'onmouseover="this.src=\'' + overlay_path.strip().lstrip(
            ) + '\';"'
            res += 'onmouseout="this.src=\'' + img_path.strip().lstrip(
            ) + '\';"'
        res += '/>'
        return res

    def addImg(self,
               img_path,
               width=400,
               imsize=None,
               overlay_path=None,
               scale=None,
               out=None):
        # bboxes must be a list of [x,y,w,h] (i.e. a list of lists)
        # imsize is the natural size of image at img_path.. used for putting bboxes, not required otherwise
        # even if it's not provided, I'll try to figure it out -- using the typical use cases of this software
        # overlay_path is image I want to show on mouseover
        assert osp.exists(img_path), img_path
        self.htmlCode += self.imgToHTML(
            osp.relpath(img_path, out), width, overlay_path)

    def addTxt(self, txt):
        if self.htmlCode:  # not empty
            self.htmlCode += '<br />'
        self.htmlCode += str(txt)

    def getHTML(self):
        return self.htmlCode

    def setIsHeader(self):
        self.isHeader = True

    def setDrawCheck(self):
        self.drawBorderColor = 'green'

    def setDrawUnCheck(self):
        self.drawBorderColor = 'red'

    def setDrawBorderColor(self, color):
        self.drawBorderColor = color

    @staticmethod
    def getImSize(impath):
        im = Image.open(impath)
        return im.size


class TableRow:

    def __init__(self, isHeader=False, rno=-1):
        self.isHeader = isHeader
        self.elements = []
        self.rno = rno

    def addElement(self, element):
        self.elements.append(element)

    def getHTML(self):
        html = '<tr>'
        if self.rno >= 0:
            html += '<td><a href="#' + str(self.rno) + '">' + str(
                self.rno) + '</a>'
            html += '<a name=' + str(self.rno) + '></a></td>'
        for e in self.elements:
            if self.isHeader or e.isHeader:
                elTag = 'th'
            else:
                elTag = 'td'
            html += '<%s>' % elTag + e.getHTML() + '</%s>' % elTag
        html += '</tr>'
        return html


class Table:

    def __init__(self, rows=[]):
        self.rows = [row for row in rows if not row.isHeader]
        self.headerRows = [row for row in rows if row.isHeader]

    def addRow(self, row):
        if not row.isHeader:
            self.rows.append(row)
        else:
            self.headerRows.append(row)

    def getHTML(self, ):
        html = '<table border=1 id="data">'
        for r in self.headerRows + self.rows:
            html += r.getHTML()
        html += '</table>'
        return html

    def countRows(self):
        return len(self.rows)


class TableWriter:

    def __init__(self,
                 table,
                 outputdir,
                 rowsPerPage=20,
                 pgListBreak=20,
                 desc=''):
        self.outputdir = outputdir
        self.rowsPerPage = rowsPerPage
        self.table = table
        self.pgListBreak = pgListBreak
        self.desc = desc

    def write(self):
        os.makedirs(self.outputdir, exist_ok=True)
        nRows = self.table.countRows()
        pgCounter = 1
        for i in range(0, nRows, self.rowsPerPage):
            rowsSubset = self.table.rows[i:i + self.rowsPerPage]
            t = Table(self.table.headerRows + rowsSubset)
            f = open(
                os.path.join(self.outputdir,
                             str(pgCounter) + '.html'), 'w')
            pgLinks = self.getPageLinks(
                int(math.ceil(nRows * 1.0 / self.rowsPerPage)), pgCounter,
                self.pgListBreak)

            f.write(pgLinks)
            f.write('<p>' + self.desc + '</p>')
            f.write(t.getHTML())
            f.write(pgLinks)
            f.close()
            pgCounter += 1

    @staticmethod
    def getPageLinks(nPages, curPage, pgListBreak):
        if nPages < 2:
            return ''
        links = ''
        for i in range(1, nPages + 1):
            if not i == curPage:
                links += '<a href="' + str(i) + '.html">' + str(
                    i) + '</a>&nbsp'
            else:
                links += str(i) + '&nbsp'
            if (i % pgListBreak == 0):
                links += '<br />'
        return '\n' + links + '\n'

def scandir(dir_path, suffix=None, recursive=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str | obj:`Path`): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.

    Returns:
        A generator for all the interested files with relative pathes.
    """
    if isinstance(dir_path, (str, Path)):
        dir_path = str(dir_path)
    else:
        raise TypeError('"dir_path" must be a string or Path object')

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if entry.name.startswith('.'):
                continue
            if not entry.name.startswith('.') and entry.is_file():
                rel_path = osp.relpath(entry.path, root)
                if suffix is None:
                    yield rel_path
                elif rel_path.endswith(suffix):
                    yield rel_path
            else:
                if recursive:
                    yield from _scandir(
                        entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate image table')
    parser.add_argument('out', help='out put directory')
    parser.add_argument('input', help='input_directory, should under out')
    parser.add_argument('--row-per-page', '-r', type=int, help='number of rows per page')
    args = parser.parse_args()
    return args

"""
├── vis_folder
│   ├── image_id
│   │   ├── block_name
│   │   │   ├── img.jpg
│   │   │   ├── attn-head0.jpg
|   |   |   ├── mask_th0.6_head0.jpg
"""

def file_cmp(f1, f2):
    # if 'img' in f1:
    #     return True
    # if 'img' in f2:
    #     return False
    return f1 > f2

def main():
    table = Table()
    args = parse_args()

    files = list(sorted(scandir(args.input, suffix='.jpg', recursive=True)))
    image_ids = set(osp.dirname(osp.dirname(f)) for f in files)
    dirs = sorted(list(set(osp.dirname(f) for f in files)))
    # files_under_dirs = {d: sorted(list(scandir(osp.join(args.input, d)))) for d in dirs}
    files_under_dirs = {d: sorted([f for f in scandir(osp.join(args.input, d)) if 'img' in f or 'bind' in f]) for d in dirs}
    num_vis_per_img = args.row_per_page or len(dirs)//len(image_ids)
    for d in files_under_dirs:
        files_under_dirs[d] = [osp.join(args.input, d, 'img.jpg')] + files_under_dirs[d]

    # images in one row
    num_cols = max(len(fs) for fs in files_under_dirs.values())
    num_rows = len(dirs)
    print(f'num cols: {num_cols}, num row: {num_rows}')

    # Header
    # header = TableRow(isHeader=True)
    # for i in range(num_cols + 1):
    #     e = Element()
    #     if i == 0:
    #         e.addTxt('index')
    #     else:
    #         e.addTxt(osp.basename(dirs[i - 1]))
    #     header.addElement(e)
    # table.addRow(header)

    for i in range(num_rows):
        # if i % 5 == 0:
        #     info_r = TableRow(rno=i)
        #     for ii in range(num_cols):
        #         e = Element()
        #         e.addTxt(osp.basename(dirs[ii]))
        #         info_r.addElement(e)
        #     table.addRow(info_r)

        # header = TableRow(isHeader=False)
        # e = Element()
        # e.addTxt(osp.basename(dirs[i]))
        # header.addElement(e)
        # table.addRow(header)
        # r = TableRow(rno=i)
        r = TableRow()
        # for j in range(num_cols):
        #     e = Element()
        #     if isinstance(files[j][i], list):
        #         # input/dir/img_dir/xxx
        #         for _ in files[j][i]:
        #             e.addImg(_, out=args.out)
        #     else:
        #         # input/dir/xxx
        #         e.addImg(files[j][i], out=args.out)
        #     r.addElement(e)
        e = Element()
        e.addTxt(osp.basename(dirs[i]))
        r.addElement(e)
        e = Element()
        for f in files_under_dirs[dirs[i]]:
            e = Element()
            e.addImg(osp.join(args.input, dirs[i], f), out=args.out)
            r.addElement(e)
        table.addRow(r)

    writer = TableWriter(table, args.out, rowsPerPage=num_vis_per_img, pgListBreak=num_vis_per_img)
    writer.write()


if __name__ == '__main__':
    main()
