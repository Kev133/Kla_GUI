import PyQt5.QtWidgets as qtw
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from PyQt5 import QtCore
import glob
import PyQt5.QtGui as qtg
from PyQt5.QtWidgets import QFileDialog, QPushButton,QStyle,\
    QPlainTextEdit,QMessageBox,QApplication,QRadioButton,QHBoxLayout,QGroupBox, QTableWidget,\
    QTableWidgetItem,QLineEdit,QFormLayout
import kla_calculation
import time
import pandas as pd
import openpyxl

class Hlavni(qtw.QMainWindow):
    def __init__(self):
        qtw.QMainWindow.__init__(self)
        # definovani vlastnosti pro program
        self.directory = None
        self.namerene = None
        self.char_sondy = None
        self.model_valuesN = []
        self.namereneN = []
        self.plot_info = False
        self.results_info = False
        self.radio_button_value= 1
        self.paramy = [0.99,0.03]
        self.font_size = 11

        self.init_ui()


    def init_ui(self):
        self.setWindowTitle(" Program for evaluation of oxygen probe data")
        self.setFont(qtg.QFont("Arial",self.font_size))
        self.setGeometry(200, 200, 920, 630)
        self.setWindowIcon(qtg.QIcon('kla_icon.png'))
        self.setStyleSheet("")
        #directory button
        self.button1 = QPushButton(" Add directory", self)
        self.button1.setGeometry(90, 25, 160, 50)
        self.button1.clicked.connect(self.find_files)
        folder = QStyle.SP_DirOpenIcon
        icon = self.style().standardIcon(folder)
        self.button1.setIcon(icon)
        #calculate button
        self.button2 = QPushButton("Calculate", self)
        self.button2.setGeometry(65, 470, 210, 50)
        self.button2.clicked.connect(self.evaluate)

        #radiobutton minimize methods
        self.group_box = QGroupBox(self)
        self.group_box.setTitle("Optimization method")
        self.group_box.setFont(qtg.QFont("Arial",self.font_size))
        self.group_box.setGeometry(30,220,280,100)
        self.hbox_layout = QHBoxLayout()
        self.opt_min1 = QRadioButton("Nelder-Mead  ", self)
        self.opt_min2 = QRadioButton("COBYLA", self)
        self.opt_min3 = QRadioButton("Powell", self)
        self.opt_min1.clicked.connect(self.nelder)
        self.opt_min2.clicked.connect(self.bfgs)
        self.opt_min3.clicked.connect(self.powell)
        self.opt_min1.setChecked(True)
        self.hbox_layout.addWidget(self.opt_min1)
        self.hbox_layout.addWidget(self.opt_min2)
        self.hbox_layout.addWidget(self.opt_min3)
        self.group_box.setLayout(self.hbox_layout)
        self.group_box.setFont(qtg.QFont("Arial", self.font_size))
        # radiobutton save plot
        self.plot_box = QGroupBox(self)
        self.plot_box.setTitle("Save plots")
        self.plot_box.setFont(qtg.QFont("Arial", self.font_size))
        self.plot_box.setGeometry(30,100,130,100)
        self.hbox_plot = QHBoxLayout()
        self.plot_box.setLayout(self.hbox_plot)
        self.plot_radio_yes= QRadioButton("Yes", self)
        self.plot_radio_no = QRadioButton("No", self)
        self.hbox_plot.addWidget(self.plot_radio_yes)
        self.hbox_plot.addWidget(self.plot_radio_no)
        self.plot_radio_yes.clicked.connect(lambda: self.save_plot(True))
        self.plot_radio_no.clicked.connect(lambda: self.save_plot(False))
        self.plot_radio_no.setChecked(True)
        self.plot_box.setFont(qtg.QFont("Arial", self.font_size))
        #radiobutton save excel
        self.excel_box = QGroupBox(self)
        self.excel_box.setTitle("Save results")
        self.excel_box.setFont(qtg.QFont("Arial", self.font_size))
        self.excel_box.setGeometry(170,100,130,100)
        self.hbox_excel = QHBoxLayout()
        self.excel_box.setLayout(self.hbox_excel)
        self.excel_radio_yes = QRadioButton("Yes", self)
        self.excel_radio_no = QRadioButton("No", self)
        self.hbox_excel.addWidget(self.excel_radio_yes)
        self.hbox_excel.addWidget(self.excel_radio_no)
        self.excel_radio_yes.clicked.connect(lambda: self.save_result(True))
        self.excel_radio_no.clicked.connect(lambda: self.save_result(False))
        self.excel_radio_no.setChecked(True)
        self.excel_box.setFont(qtg.QFont("Arial", self.font_size))
        # Set limits (paramy)
        self.e1 = QLineEdit(self)
        self.e2 = QLineEdit(self)
        self.e1.setText("0.99")
        self.e1.setFixedWidth(55)
        self.e2.setFixedWidth(55)
        self.e2.setText("0.03")
        frame = QGroupBox(self)
        frame.setGeometry(80, 340, 180, 100)
        layout = QFormLayout()
        layout.setVerticalSpacing(20)
        layout.addRow("Upper limit", self.e1)
        layout.addRow("Lower limit", self.e2)
        frame.setLayout(layout)
        frame.setTitle("Set limits")
        #info for user
        self.info_for_user = QPlainTextEdit("                 Information panel\n\n"
            "Welcome, this program calculates kla from experimental data.\n\n", self)
        self.info_for_user.setReadOnly(True)
        self.info_for_user.setGeometry(335, 30, 270, 490)

        #Table
        self.table = QTableWidget(self)
        self.table.setGeometry(635, 30, 245, 570)
        self.table.setRowCount(18)
        self.table.setColumnCount(2)
        self.labels = ["Exp. name","kla"]
        self.table.setHorizontalHeaderLabels(self.labels)

        self.clear_button = QPushButton('Clear', self)
        self.clear_button.clicked.connect(self.clear_text_table)
        self.clear_button.setGeometry(410,550,120,50)
        self.show()
    def get_paramy(self):
        self.paramy[0] = float(self.e1.text())
        self.paramy[1] = float(self.e2.text())
        return self.paramy
    def save_result(self,arg):
        self.results_info = arg

    def save_plot(self, arg):
        self.plot_info = arg

    def clear_text_table(self):
        self.table.clear()
        self.info_for_user.clear()
        self.table.setHorizontalHeaderLabels(self.labels)
        self.info_for_user.setPlainText(f"                 Information panel\n\n")
    def keyPressEvent(self, event):
        """Allows the user to copy the values from the Qtable,
        somehow this is not a built in function"""
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

    #funkce pro vyber optimalizacni metody, cislo je pak passed do kla_evaluator
    def nelder(self):
        self.radio_button_value = 1
    def bfgs(self):
        self.radio_button_value = 2
    def powell(self):
        self.radio_button_value = 3


    def find_files(self):
        directory = QFileDialog.getExistingDirectory(self)
        # if name is not empty, which means the user picked a directory, the below code will execute
        if directory:
            self.directory = directory
            self.info_for_user.appendPlainText("The necessary files are in the folder\n\n"+directory+"\n")


            self.dtm_files = glob.glob(self.directory + "/*.dtm")
            self.info_for_user.appendPlainText(f"There are {len(self.dtm_files)} .dtm files in this folder and ")
            self.dta_files = glob.glob(self.directory + "/*.DTA")
            self.konstant=None
            for file in self.dta_files:
                if "konstant" in file.lower():  # tries to find a file that has "konstant" or "KONSTANT" in its name
                    self.konstant = file  # after the file is found, it is called konstant
                    self.info_for_user.appendPlainText(f"konstant.dta is also present.\n")
                    break
            if not self.konstant:
                self.info_for_user.appendPlainText(f"konstant.dta is NOT present.")
            QtCore.QCoreApplication.processEvents()


    def evaluate(self):
        if self.directory == None:
            QMessageBox.warning(self,"Warning","A file directory has not been added")
            return
        if len(self.dtm_files)==0:
            QMessageBox.warning(self, "Warning", "There are no .dtm files in this folder")
            return
        self.paramy = self.get_paramy()
        print("Evaluation of kla has started")
        self.start_time=time.time() # spustí se čas


        self.kla_list = []
        self.rows_excel = []
        try:
            self.call_thread()
        except Exception as e:
            print(str(e))
    def call_thread(self):
        self.info_for_user.appendPlainText("Calculation has started\n")
        # TODO az na .start() presunout do __init__, zavolat z button rovnou self.worker.start() ?? jak??
        self.worker = WorkerThread(self.dtm_files,self.konstant, self.directory, self.radio_button_value, self.plot_info, self.paramy)
        self.worker.start()
        self.worker.update_signal_name.connect(self.update_info_name)
        self.worker.update_signal_kla.connect(self.update_info_kla)
        self.worker.finish_signal.connect(self.get_dict)

    def update_info_name(self,measurement_name):
        self.info_for_user.appendPlainText(measurement_name)
    def update_info_kla(self,kla):
        self.info_for_user.appendPlainText(f"Found kla {round(kla, 6)} s\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT ONE}\n")
    def get_dict(self, lists):

        self.kla_list = lists["kla"]
        self.rows_excel = lists["names"]
        self.header = lists["header"]
        self.update_GUI_table(self.kla_list, self.rows_excel)

        self.end_time=time.time() #vypne se čas
        #upravy pro zjisteni jak dlouho vypocet trval
        self.elapsed_time=self.end_time-self.start_time
        self.info_for_user.appendPlainText(f"Time for calculations: {round(self.elapsed_time,1)} s\n")

        if self.results_info == True:
            self.save_results(lists)
        self.info_for_user.appendPlainText(
                "Finished, if you have saved your data, you can use the clear button and start a new evaluation\n")

    def save_results(self, lists):
        self.excel_name = "/"+lists["header"]+" results.xlsx"
        indexes= lists.pop("names")
        lists.pop("header")
        vyp_excel = pd.DataFrame(data=lists, index=indexes)
        with pd.ExcelWriter(self.directory + self.excel_name) as writer:
            vyp_excel.to_excel(writer, sheet_name="results")
        self.info_for_user.appendPlainText("You will find your kla values in an excel file in the directory of the experiment\n")

    def update_GUI_table(self,kla_list,list_excel):
        for i in range (0,len(self.dtm_files)):
            self.table.setItem(i,0,QTableWidgetItem(str(list_excel[i])))
            self.table.setItem(i,1,QTableWidgetItem(str(round(kla_list[i],8))))

class WorkerThread(QtCore.QThread):
    update_signal_kla = QtCore.pyqtSignal(float)
    update_signal_name = QtCore.pyqtSignal(str)
    finish_signal = QtCore.pyqtSignal(dict)
    def __init__(self,dtm_files,konstant_file,name,choice,plot_info,paramy):
        super().__init__()
        self.dtm_files = dtm_files
        self.name = name
        self.choice_radiobutton = choice
        self.plot_info = plot_info
        self.paramy = paramy
        self.konstant_file = konstant_file


    def run(self):
        kla_values = []
        exp_names = []
        gas_in_list = []
        agitator_frequency_list = []
        gas_hold_up_list = []
        for dtm_file in self.dtm_files:


            results_dict = kla_calculation.main_function(dtm_file, self.konstant_file,self.name, self.choice_radiobutton,self.plot_info,self.paramy)
            kla = results_dict[0]
            measurement_name = results_dict[1]
            header = results_dict[2]
            gas_in = results_dict[3]
            agitator_frequency = results_dict[4]
            gas_hold_up = results_dict[5]
            self.update_signal_name.emit(measurement_name)
            self.update_signal_kla.emit(kla)

            exp_names.append(measurement_name)
            kla_values.append(kla)
            gas_in_list.append(gas_in)
            gas_hold_up_list.append(gas_hold_up)
            agitator_frequency_list.append(agitator_frequency)
        lists_dict = { "names":exp_names,"Vg.":gas_in_list,"hold-up":gas_hold_up_list,
                 "f mix":agitator_frequency_list,"kla":kla_values,"header":header}
        self.finish_signal.emit(lists_dict)
def main():

    app = qtw.QApplication([])
    gui = Hlavni()
    app.exec_()

if __name__=="__main__":
    main()
