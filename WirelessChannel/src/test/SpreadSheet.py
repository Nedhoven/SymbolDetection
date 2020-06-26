
import xlwt

import src.main.IO as IO


def test_output():
    wb = xlwt.Workbook()
    sheet1 = wb.add_sheet('Channel Use #' + str(1))
    sheet1.write(0, 0, 'user id')
    sheet1.write(0, 1, 'channel tap')
    sheet1.write(0, 2, 'transmitter id')
    sheet1.write(0, 3, 'channel')
    sheet1.write(0, 4, 'time lot')
    sheet1.write(0, 5, 'input')
    sheet1.write(0, 6, 'desired')
    sheet1.write(0, 7, 'output')
    d = IO.DataWriter()
    d.data_to_xl(wb, 'testing')
    return
