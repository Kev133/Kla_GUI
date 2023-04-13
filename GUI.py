import PyQt5.QtWidgets as qtw
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
import numpy as np
from PyQt5 import QtCore
import glob
import PyQt5.QtGui as qtg
from PyQt5.QtWidgets import QWidget, QLabel, QFileDialog, QPushButton, QDialog,QFontDialog,QStyle,\
    QPlainTextEdit,QMessageBox,QApplication,QRadioButton,QHBoxLayout,QGroupBox,QVBoxLayout, QTableWidget,\
    QTableWidgetItem,QAbstractItemView
import sys
import threading
import kla_evalueator
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import time
import pandas as pd
import openpyxl

class Hlavni(qtw.QMainWindow):
    def __init__(self):
        qtw.QMainWindow.__init__(self)
        # definovani vlastnosti pro program
        self.name = None
        self.namerene = None
        self.char_sondy = None
        self.model_valuesN = []
        self.namereneN = []
        self.plot_info = False
        self.radio_button_value= 1
        self.init_ui()

    def init_ui(self):
        # graficke veci

        self.setWindowTitle("  KLA Evaluator 2000")
        self.setFont(qtg.QFont("Arial",11))
        self.setGeometry(200, 200, 900, 500)
        self.setWindowIcon(qtg.QIcon('kla_icon.png'))
        self.setStyleSheet("")

        self.button1 = QPushButton(" Add directory", self)
        self.button1.setGeometry(430, 20, 130, 40)
        self.button1.clicked.connect(self.pruzkum)
        # ikona u přidat adresar
        ikonka = QStyle.SP_DirOpenIcon
        icon = self.style().standardIcon(ikonka)

        self.button1.setIcon(icon)

        # self.button_help = QPushButton("Help", self)
        # self.button_help.clicked.connect(self.help_okno)
        # self.button_help.setGeometry(850, 0, 50, 30)

        self.button2 = QPushButton("Calculate", self)
        self.button2.setGeometry(430, 65, 130, 40)
        self.button2.clicked.connect(self.evaluate)

        #radiobutton minimize methods
        self.group_box = QGroupBox(self)
        self.group_box.setTitle("Optimization method")
        self.group_box.setFont(qtg.QFont("Arial",11))
        # vbox = QVBoxLayout()
        # vbox.addWidget(self.group_box)
        self.group_box.setGeometry(70,10,330,100)
        self.hbox_layout = QHBoxLayout()
        # radiobutton save plot
        self.plot_box = QGroupBox(self)
        self.plot_box.setTitle("Save plots")
        self.plot_box.setFont(qtg.QFont("Arial", 11))

        self.plot_box.setGeometry(300, 130, 150, 100)
        self.hbox_plot = QHBoxLayout()
        self.plot_box.setLayout(self.hbox_plot)
        self.opt_min1 = QRadioButton("Nelder-Mead  ",self)
        self.opt_min2 = QRadioButton("COBYLA",self)
        self.opt_min3 = QRadioButton("Powell",self)
        self.opt_min1.setChecked(True)
        self.hbox_layout.addWidget(self.opt_min1)
        self.hbox_layout.addWidget(self.opt_min2)
        self.hbox_layout.addWidget(self.opt_min3)

        self.radio_yes= QRadioButton("Yes", self)
        self.radio_no = QRadioButton("No", self)
        self.hbox_plot.addWidget(self.radio_yes)
        self.hbox_plot.addWidget(self.radio_no)
        self.radio_no.setChecked(True)
        self.radio_yes.clicked.connect(self.save_plot)



        self.opt_min1.clicked.connect(self.nelder)
        self.opt_min2.clicked.connect(self.bfgs)
        self.opt_min3.clicked.connect(self.powell)

        self.group_box.setLayout(self.hbox_layout)
        self.group_box.setFont(qtg.QFont("Arial", 11))

        self.info_for_user = QPlainTextEdit("Welcome, this program calculates kla from experimental data.\n\n", self)
        self.info_for_user.setGeometry(580, 20, 270, 440)
        self.info_for_user.setReadOnly(True)

        self.table = QTableWidget(self)
        self.table.setGeometry(20, 130, 250, 330)
        self.table.setRowCount(18)
        self.table.setColumnCount(2)
        #TODO does this do anything?
        self.table.setSelectionMode(QAbstractItemView.ContiguousSelection)
        #TODO lol to mam tady hard-coded jo?


        self.show()
    def save_plot(self):
        self.plot_info = True

    def keyPressEvent(self, event):
        try:
            if event.matches(qtg.QKeySequence.Copy):
                selection = self.table.selectedRanges()
                if selection:
                    top_row = selection[0].topRow()
                    bottom_row = selection[0].bottomRow()
                    left_col = selection[0].leftColumn()
                    right_col = selection[0].rightColumn()

                    text = ""
                    for row in range(top_row, bottom_row + 1):
                        row_text = ""
                        for col in range(left_col, right_col + 1):
                            item = self.table.item(row, col)
                            if item is not None:
                                row_text += item.text()
                            row_text += "\t"
                        row_text = row_text.rstrip("\t") + "\n"
                        text += row_text

                    QApplication.clipboard().setText(text)
            else:
                super().keyPressEvent(event)
        except Exception as e:
            print(str(e))

        #zavede cistou canvas pro graf
        # self.figure = plt.figure()
        # self.canvas = FigureCanvas(self.figure)
        # self.canvas.setParent(self)
        # self.canvas.setGeometry(20, 120, 550, 340)
        # self.fig_ax = self.figure.add_subplot(111)
        # self.figure.set_tight_layout(True)


    #funkce pro vyber optimalizacni metody, cislo je pak passed do kla_evaluator
    def nelder(self):
        self.radio_button_value = 1
    def bfgs(self):
        self.radio_button_value = 2
    def powell(self):
        self.radio_button_value = 3


    def pruzkum(self):
        name = QFileDialog.getExistingDirectory(self)
        # if name is not empty, which means the user picked a directory, the below code will execute
        if name:
            self.name = name
            self.info_for_user.appendPlainText("The necessary files are in the folder\n\n"+name+"\n")


            self.dtm_files = glob.glob(self.name + "/*.dtm")
            self.info_for_user.appendPlainText(f"There are {len(self.dtm_files)} .dtm files in this folder and ")
            self.dta_files = glob.glob(self.name+"/*.DTA")
            konstant=None
            for file in self.dta_files:
                if "konstant" in file.lower():  # tries to find a file that has "konstant" or "KONSTANT" in its name
                    konstant = file  # after the file is found, it is called konstant
                    self.info_for_user.appendPlainText(f"konstant.dta is also present.\n")
                    break
            if not konstant:
                self.info_for_user.appendPlainText(f"konstant.dta is NOT present.")
            QtCore.QCoreApplication.processEvents()





    def evaluate(self):
        if len(self.dtm_files)==0:
            QMessageBox.warning(self, "Warning", "There are no .dtm files in this folder")
        if self.name == None:
            QMessageBox.warning(self,"Warning","A file directory has not been added")

        print("Evaluation of kla has started")
        self.start_time=time.time() # spustí se čas


        self.kla_list = []
        self.rows_excel = []
        try:
            self.call_thread()
        except Exception as e:
            print(str(e))
    def call_thread(self):
        # TODO az na .start() presunout do __init__, zavolat z button rovnou self.worker.start()
        self.worker = WorkerThread(self.dtm_files,self.name,self.radio_button_value,self.plot_info)
        self.worker.start()
        self.worker.update_signal_name.connect(self.update_info_name)
        self.worker.update_signal_kla.connect(self.update_info_kla)
        self.worker.finish_signal.connect(self.get_dict)
        # try:
        #     for i in range (0,len(self.dtm_files)):
        #         kla = [0]
        #         data = kla_evalueator.main_function(i,self.name,self.radio_button_value)
        #         kla[0]=data[0]
        #
        #         measurement_name = data[1]
        #
        #
        #         self.info_for_user.appendPlainText(measurement_name)
        #         self.info_for_user.appendPlainText(f"Found kla {round(kla[0],6)}")
        #         QApplication.processEvents()
        #
        #         rows_excel.append(measurement_name)
        #         kla_list.append(kla)
        #
        #     self.ulozeni_dat(kla_list,rows_excel)
        #     self.update_GUI_table(kla_list,rows_excel)
        #     print(rows_excel)
        # except Exception as e:
        #     print(str(e))
    def update_info_name(self,measurement_name):
        self.info_for_user.appendPlainText(measurement_name)
    def update_info_kla(self,kla):
        self.info_for_user.appendPlainText(f"Found kla {round(kla, 6)} s\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT ONE}\n")
    def get_dict(self,dict):

        self.kla_list = dict["kla"]
        self.rows_excel = dict["names"]


        self.ulozeni_dat(self.kla_list, self.rows_excel)
        self.update_GUI_table(self.kla_list, self.rows_excel)
        print(self.rows_excel)
        self.end_time=time.time() #vypne se čas
        #upravy pro zjisteni jak dlouho vypocet trval
        self.elapsed_time=self.end_time-self.start_time

        # #graficke veci, ktere asi ve vypoctu ani nemusi byt
        self.info_for_user.appendPlainText(f"Time for calculations: {round(self.elapsed_time,1)} s\n")
        # QtCore.QCoreApplication.processEvents()
        #
        # #self.info_for_user.appendPlainText(f"Hodnota kLa je: {round(self.kla,7)} s\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT ONE}\n")
        # #self.plotni_to()
        # QMessageBox.information(self,f"Čas výpočtu: {round(self.elapsed_time,2)} s")
    def plotni_to(self):
        tau = np.linspace(0, 155, num=3101)

        self.fig_ax.plot(tau, self.namerene)
        self.fig_ax.plot(tau,self.char_sondy)
        self.fig_ax.plot(tau,self.model_konv)

        self.fig_ax.set_xlabel("čas [s]")
        self.fig_ax.set_ylabel("x0\N{SUBSCRIPT TWO} normalizované")
        self.fig_ax.margins(x=0)
        self.fig_ax.legend(["naměřené hodnoty","impulzní charakteristika","teoretický průběh koncentrace"])
        self.canvas.draw()

        # add toolbar
        self.addToolBar(Qt.BottomToolBarArea, NavigationToolbar(self.canvas, self))


    def ulozeni_dat(self,kla_list,list_excel):
        self.excel_name = "kla_results.xlsx"
        vyp_excel = pd.DataFrame(kla_list, index=list_excel, columns=["kLa"])
        with pd.ExcelWriter(self.excel_name) as writer:
            vyp_excel.to_excel(writer, sheet_name="results")
        self.info_for_user.appendPlainText("FINISHED, you will find the data in the excel file")
    def update_GUI_table(self,kla_list,list_excel):
        for i in range (0,len(self.dtm_files)):
                self.table.setItem(i,0,QTableWidgetItem(str(list_excel[i])))
                self.table.setItem(i,1,QTableWidgetItem(str(round(kla_list[i][0],8))))

class WorkerThread(QtCore.QThread):
    update_signal_kla = QtCore.pyqtSignal(float)
    update_signal_name = QtCore.pyqtSignal(str)
    finish_signal = QtCore.pyqtSignal(dict)
    def __init__(self,dtm_files,name,choice,plot_info):
        super().__init__()
        self.dtm_files = dtm_files
        self.name = name
        self.choice_radiobutton = choice
        self.plot_info = plot_info

    def run(self):
        try:
            self.kla_list = []
            self.rows_excel = []
            for i in range(0, len(self.dtm_files)):
                kla = [0]
                #TODO zde poslat jen jmeno ne i, loop pres jmena ne pres i
                data = kla_evalueator.main_function(i, self.name, self.choice_radiobutton,self.plot_info)
                measurement_name = data[1]
                kla[0] = data[0]
                self.update_signal_name.emit(measurement_name)
                self.update_signal_kla.emit(kla[0])

                self.rows_excel.append(measurement_name)
                self.kla_list.append(kla)
            lists = {"kla":self.kla_list,"names":self.rows_excel}
            self.finish_signal.emit(lists)
        except Exception as e:
            print(str(e))

            #self.info_for_user.appendPlainText(measurement_name)
            #self.info_for_user.appendPlainText(f"Found kla {round(kla[0], 6)}")

def main():

    app = qtw.QApplication([])
    gui = Hlavni()
    app.exec_()

if __name__=="__main__":
    main()
