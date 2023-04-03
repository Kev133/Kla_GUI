import PyQt5.QtWidgets as qtw
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
import numpy as np
from PyQt5 import QtCore
import glob
import PyQt5.QtGui as qtg
from PyQt5.QtWidgets import QWidget, QLabel, QFileDialog, QPushButton, QDialog,QFontDialog,QStyle,\
    QPlainTextEdit,QMessageBox,QApplication,QRadioButton,QHBoxLayout,QGroupBox,QVBoxLayout, QTableWidget,QTableWidgetItem
import sys


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
        self.init_ui()

    def init_ui(self):
        # graficke veci

        self.setWindowTitle("  Vyhodnocení dat z kyslíkové sondy")
        self.setFont(qtg.QFont("Arial",11))
        self.setGeometry(200, 200, 900, 500)
        self.setWindowIcon(qtg.QIcon('kla_icon.png'))
        self.setStyleSheet("")

        self.button1 = QPushButton(" Přidat adresář", self)
        self.button1.setGeometry(430, 20, 130, 40)
        self.button1.clicked.connect(self.pruzkum)
        # ikona u přidat adresar
        ikonka = QStyle.SP_DirOpenIcon
        icon = self.style().standardIcon(ikonka)

        self.button1.setIcon(icon)

        # self.button_help = QPushButton("Help", self)
        # self.button_help.clicked.connect(self.help_okno)
        # self.button_help.setGeometry(850, 0, 50, 30)

        self.button2 = QPushButton("Provést výpočet", self)
        self.button2.setGeometry(430, 65, 130, 40)
        self.button2.clicked.connect(self.evaluate)

        #radiobutton
        self.group_box = QGroupBox(self)
        self.group_box.setTitle("Optimalizační metoda")
        self.group_box.setFont(qtg.QFont("Arial",11))
        # vbox = QVBoxLayout()
        # vbox.addWidget(self.group_box)
        self.group_box.setGeometry(70,10,330,100)
        self.hbox_layout = QHBoxLayout ()

        self.opt_min1 = QRadioButton("Nelder-Mead  ",self)
        self.opt_min2 = QRadioButton("COBYLA",self)
        self.opt_min3 = QRadioButton("Powell",self)
        self.opt_min1.setChecked(True)

        self.hbox_layout.addWidget(self.opt_min1)
        self.hbox_layout.addWidget(self.opt_min2)
        self.hbox_layout.addWidget(self.opt_min3)

        self.opt_min1.clicked.connect(self.nelder)
        self.opt_min2.clicked.connect(self.bfgs)
        self.opt_min3.clicked.connect(self.powell)

        self.group_box.setLayout(self.hbox_layout)
        self.group_box.setFont(qtg.QFont("Arial", 11))

        self.info_for_user = QPlainTextEdit("Welcome, this program calculates kla from experimental data.\n\n", self)
        self.info_for_user.setGeometry(580, 20, 270, 440)
        self.info_for_user.setReadOnly(True)

        self.table = QTableWidget(self)
        self.table.setGeometry(20, 130, 530, 330)
        self.table.setRowCount(18)
        self.table.setColumnCount(4)
        #zavede cistou canvas pro graf
        # self.figure = plt.figure()
        # self.canvas = FigureCanvas(self.figure)
        # self.canvas.setParent(self)
        # self.canvas.setGeometry(20, 120, 550, 340)
        # self.fig_ax = self.figure.add_subplot(111)
        # self.figure.set_tight_layout(True)

        self.radio_button_value = 1

        self.show()

    #funkce pro vyber optimalizacni metody, cislo je pak passed do vypocet.Optimalizace.opt()
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
                    self.info_for_user.appendPlainText(f"konstant.dta is also present.")
                    break
            if not konstant:
                self.info_for_user.appendPlainText(f"konstant.dta is NOT present.")
            QtCore.QCoreApplication.processEvents()
        # except Exception as e:
        # #toto mi říká reálně všechny errory v programu, ne jen v části načtení dat
        #     print(str(e))
        #     self.info_for_user.appendPlainText("V této složce nejsou vhodné soubory pro výpočet.\nZvolte správnou složku.")


    def evaluate(self):
        if len(self.dtm_files)==0:
            QMessageBox.warning(self, "Warning", "There are no .dtm files in this folder")
        if self.name == None:
            QMessageBox.warning(self,"Warning","A file directory has not been added")

        print("Evaluation of kla has started")
        self.start_time=time.time() # spustí se čas

        #zde se vola druhy py soubor s vypoctem a optimalizaci
        import kla_evalueator
        kla_list = []
        rows_excel = []
        try:
            for i in range (0,len(self.dtm_files)):
                kla = [0]

                data = kla_evalueator.main_function(i,self.name,self.radio_button_value)
                kla[0]=data[0]

                measurement_name = data[1]
                self.info_for_user.appendPlainText(measurement_name)
                self.info_for_user.appendPlainText(f"Found kla {round(kla[0],6)}")
                QtCore.QCoreApplication.processEvents()
                rows_excel.append(measurement_name)
                kla_list.append(kla)

            self.ulozeni_dat(kla_list,rows_excel)
            self.update_GUI_table(kla_list,rows_excel)
            print(rows_excel)
        except Exception as e:
            print(str(e))
        #kla = [0]
        #kla[0] = kla_evalueator.opt(1)[0]
        #print(kla[0])


        self.end_time=time.time() #vypne se čas
        #upravy pro zjisteni jak dlouho vypocet trval
        self.elapsed_time=self.end_time-self.start_time

        # #graficke veci, ktere asi ve vypoctu ani nemusi byt
        # self.info_for_user.appendPlainText(f"Čas výpočtu: {round(self.elapsed_time,2)} s\n")
        # QtCore.QCoreApplication.processEvents()
        #
        # #self.info_for_user.appendPlainText(f"Hodnota kLa je: {round(self.kla,7)} s\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT ONE}\n")
        # #self.plotni_to()
        # QMessageBox.information(self,f"Čas výpočtu: {round(self.elapsed_time,2)} s")
    def plotni_to(self):
        tau = np.linspace(0, 155, num=3101)
        # TODO: dotaz!
        #je dulezite to mit pod self.model_konv?
        #musim u všeho dávat self? U ceho bych nemusel, viz Impulzovka
        #nahradit pridanim do zavorky k fci plotni to


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
        print("FINISHED, you will find the data in the excel file")
        self.info_for_user.appendPlainText("Výsledek je uložen v excel souboru kla_vysledky.xlsx")
    def update_GUI_table(self,kla_list,list_excel):
        print(list_excel[1])
        print(kla_list[0])
        for i in range (0,len(self.dtm_files)):
                self.table.setItem(i,0,QTableWidgetItem(str(list_excel[i])))
                self.table.setItem(i,1,QTableWidgetItem(str(round(kla_list[i][0],6))))

        # path = self.name + "/" + self.excel_name
        # workbook = openpyxl.load_workbook(path)
        # sheet = workbook.active
        # list_values = list(sheet.values)
def main():

    app = qtw.QApplication([])
    gui = Hlavni()
    app.exec_()

if __name__=="__main__":
    main()
