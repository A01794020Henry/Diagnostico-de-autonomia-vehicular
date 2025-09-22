"""
Interfaz gráfica para visualización interactiva de señales CAN
============================================================

Interfaz basada en PyQt5 y PyQtGraph para visualizar señales decodificadas
de archivos BLF con capacidad de filtrado y análisis temporal.

Autor: Sistema de diagnóstico de autonomía vehicular
Fecha: 2025
"""

import sys
import os
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QComboBox, QListWidget, QSplitter,
                             QGroupBox, QLabel, QLineEdit, QCheckBox, QProgressBar,
                             QFileDialog, QMessageBox, QAbstractItemView, QTabWidget,
                             QTableWidget, QTableWidgetItem, QHeaderView, QTextEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor, QPalette
import pyqtgraph as pg
from datetime import datetime, timedelta
import logging

# Importar nuestro procesador
from ProcessorBLF_v2 import ProcessorBLF

# Configurar PyQtGraph
pg.setConfigOptions(antialias=True)
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

logger = logging.getLogger(__name__)

class ProcessingThread(QThread):
    """
    Hilo separado para procesamiento de archivos BLF sin bloquear la interfaz.
    """
    progress_updated = pyqtSignal(int, str)
    processing_finished = pyqtSignal(pd.DataFrame)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, blf_directory, dbc_path=None):
        super().__init__()
        self.blf_directory = blf_directory
        self.dbc_path = dbc_path
        self.processor = ProcessorBLF()
    
    def run(self):
        try:
            self.progress_updated.emit(10, "Inicializando procesador...")
            
            if self.dbc_path:
                self.progress_updated.emit(20, "Cargando archivo DBC...")
                if not self.processor.load_dbc(self.dbc_path):
                    self.error_occurred.emit("Error cargando archivo DBC")
                    return
            
            self.progress_updated.emit(30, "Buscando archivos BLF...")
            blf_files = self.processor.find_blf_files(self.blf_directory)
            
            if not blf_files:
                self.error_occurred.emit("No se encontraron archivos BLF")
                return
            
            self.progress_updated.emit(50, f"Procesando {len(blf_files)} archivos BLF...")
            unified_df = self.processor.unify_blf_files(blf_files)
            
            if unified_df.empty:
                self.error_occurred.emit("No se pudieron procesar los archivos BLF")
                return
            
            self.progress_updated.emit(80, "Decodificando mensajes...")
            decoded_df = self.processor.decode_messages(unified_df)
            
            self.progress_updated.emit(100, "Procesamiento completado")
            self.processing_finished.emit(decoded_df)
            
        except Exception as e:
            self.error_occurred.emit(f"Error durante el procesamiento: {str(e)}")

class SignalPlotWidget(QWidget):
    """
    Widget para mostrar gráficos de señales individuales.
    """
    
    def __init__(self):
        super().__init__()
        self.initUI()
        self.plot_items = {}
        
    def initUI(self):
        layout = QVBoxLayout()
        
        # Widget de gráfico principal
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Valor')
        self.plot_widget.setLabel('bottom', 'Tiempo (s)')
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.addLegend()
        
        # Configurar zoom y pan
        self.plot_widget.setMouseEnabled(x=True, y=True)
        self.plot_widget.enableAutoRange()
        
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)
    
    def plot_signal(self, df, signal_name, color=None):
        """
        Grafica una señal específica.
        """
        if df.empty:
            return
        
        # Filtrar datos para la señal específica
        signal_data = df[df['signal_name'] == signal_name].copy()
        
        if signal_data.empty:
            return
        
        # Preparar datos para el gráfico
        x_data = signal_data['timestamp'].values
        y_data = pd.to_numeric(signal_data['signal_value'], errors='coerce').values
        
        # Remover valores NaN
        valid_mask = ~np.isnan(y_data)
        x_data = x_data[valid_mask]
        y_data = y_data[valid_mask]
        
        if len(x_data) == 0:
            return
        
        # Color automático si no se especifica
        if color is None:
            colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k']
            color_idx = len(self.plot_items) % len(colors)
            color = colors[color_idx]
        
        # Crear el gráfico
        pen = pg.mkPen(color=color, width=2)
        plot_item = self.plot_widget.plot(x_data, y_data, pen=pen, name=signal_name)
        
        # Almacenar referencia
        self.plot_items[signal_name] = plot_item
        
        # Obtener unidad si está disponible
        unit = signal_data['unit'].iloc[0] if 'unit' in signal_data.columns and not signal_data['unit'].isna().all() else ''
        if unit:
            self.plot_widget.setLabel('left', f'Valor ({unit})')
    
    def clear_plots(self):
        """
        Limpia todos los gráficos.
        """
        self.plot_widget.clear()
        self.plot_items.clear()
    
    def remove_signal(self, signal_name):
        """
        Remueve una señal específica del gráfico.
        """
        if signal_name in self.plot_items:
            self.plot_widget.removeItem(self.plot_items[signal_name])
            del self.plot_items[signal_name]

class DataTableWidget(QWidget):
    """
    Widget para mostrar datos en formato tabular.
    """
    
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()
        
        # Tabla de datos
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        
        layout.addWidget(self.table)
        self.setLayout(layout)
    
    def update_table(self, df, max_rows=1000):
        """
        Actualiza la tabla con datos del DataFrame.
        """
        if df.empty:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return
        
        # Limitar número de filas para rendimiento
        display_df = df.head(max_rows)
        
        # Configurar tabla
        self.table.setRowCount(len(display_df))
        self.table.setColumnCount(len(display_df.columns))
        self.table.setHorizontalHeaderLabels(display_df.columns.tolist())
        
        # Llenar datos
        for i, (_, row) in enumerate(display_df.iterrows()):
            for j, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                self.table.setItem(i, j, item)
        
        # Ajustar ancho de columnas
        self.table.resizeColumnsToContents()
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)

class CANVisualizerGUI(QMainWindow):
    """
    Ventana principal de la aplicación de visualización CAN.
    """
    
    def __init__(self):
        super().__init__()
        self.processor = None
        self.current_data = pd.DataFrame()
        self.selected_signals = set()
        
        self.initUI()
        self.setup_connections()
    
    def initUI(self):
        """
        Inicializa la interfaz de usuario.
        """
        self.setWindowTitle("Visualizador de Señales CAN - Diagnóstico de Autonomía Vehicular")
        self.setGeometry(100, 100, 1400, 900)
        
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        main_layout = QHBoxLayout()
        
        # Panel de control (izquierda)
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, stretch=1)
        
        # Panel de visualización (derecha)
        viz_panel = self.create_visualization_panel()
        main_layout.addWidget(viz_panel, stretch=3)
        
        central_widget.setLayout(main_layout)
        
        # Barra de estado
        self.statusBar().showMessage("Listo para cargar archivos BLF")
        
        # Barra de progreso
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)
    
    def create_control_panel(self):
        """
        Crea el panel de control lateral.
        """
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Sección de carga de archivos
        file_group = QGroupBox("Carga de Archivos")
        file_layout = QVBoxLayout()
        
        self.blf_path_edit = QLineEdit()
        self.blf_path_edit.setPlaceholderText("Directorio de archivos BLF")
        self.blf_browse_btn = QPushButton("Buscar Directorio BLF")
        
        self.dbc_path_edit = QLineEdit()
        self.dbc_path_edit.setPlaceholderText("Archivo DBC (opcional)")
        self.dbc_browse_btn = QPushButton("Buscar Archivo DBC")
        
        self.process_btn = QPushButton("Procesar Archivos")
        self.process_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        
        file_layout.addWidget(QLabel("Directorio BLF:"))
        file_layout.addWidget(self.blf_path_edit)
        file_layout.addWidget(self.blf_browse_btn)
        file_layout.addWidget(QLabel("Archivo DBC:"))
        file_layout.addWidget(self.dbc_path_edit)
        file_layout.addWidget(self.dbc_browse_btn)
        file_layout.addWidget(self.process_btn)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Sección de filtros
        filter_group = QGroupBox("Filtros")
        filter_layout = QVBoxLayout()
        
        # Filtro por mensaje
        self.message_combo = QComboBox()
        self.message_combo.addItem("Todos los mensajes")
        
        # Lista de señales
        self.signals_list = QListWidget()
        self.signals_list.setSelectionMode(QAbstractItemView.MultiSelection)
        
        # Botones de control de señales
        signal_btn_layout = QHBoxLayout()
        self.add_signal_btn = QPushButton("Agregar Señal")
        self.remove_signal_btn = QPushButton("Remover Señal")
        self.clear_signals_btn = QPushButton("Limpiar Todo")
        
        signal_btn_layout.addWidget(self.add_signal_btn)
        signal_btn_layout.addWidget(self.remove_signal_btn)
        signal_btn_layout.addWidget(self.clear_signals_btn)
        
        filter_layout.addWidget(QLabel("Mensaje:"))
        filter_layout.addWidget(self.message_combo)
        filter_layout.addWidget(QLabel("Señales disponibles:"))
        filter_layout.addWidget(self.signals_list)
        filter_layout.addLayout(signal_btn_layout)
        
        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)
        
        # Sección de estadísticas
        stats_group = QGroupBox("Estadísticas")
        stats_layout = QVBoxLayout()
        
        self.stats_text = QTextEdit()
        self.stats_text.setMaximumHeight(150)
        self.stats_text.setReadOnly(True)
        
        stats_layout.addWidget(self.stats_text)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # Botones de exportación
        export_group = QGroupBox("Exportar")
        export_layout = QVBoxLayout()
        
        self.export_csv_btn = QPushButton("Exportar a CSV")
        self.export_plot_btn = QPushButton("Exportar Gráfico")
        
        export_layout.addWidget(self.export_csv_btn)
        export_layout.addWidget(self.export_plot_btn)
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel
    
    def create_visualization_panel(self):
        """
        Crea el panel de visualización principal.
        """
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Pestañas para diferentes vistas
        self.tab_widget = QTabWidget()
        
        # Pestaña de gráficos
        self.plot_widget = SignalPlotWidget()
        self.tab_widget.addTab(self.plot_widget, "Gráficos")
        
        # Pestaña de datos tabulares
        self.table_widget = DataTableWidget()
        self.tab_widget.addTab(self.table_widget, "Datos")
        
        layout.addWidget(self.tab_widget)
        panel.setLayout(layout)
        return panel
    
    def setup_connections(self):
        """
        Configura las conexiones de señales y slots.
        """
        # Botones de navegación de archivos
        self.blf_browse_btn.clicked.connect(self.browse_blf_directory)
        self.dbc_browse_btn.clicked.connect(self.browse_dbc_file)
        
        # Procesamiento
        self.process_btn.clicked.connect(self.process_files)
        
        # Filtros y controles
        self.message_combo.currentTextChanged.connect(self.update_signals_list)
        self.add_signal_btn.clicked.connect(self.add_selected_signals)
        self.remove_signal_btn.clicked.connect(self.remove_selected_signals)
        self.clear_signals_btn.clicked.connect(self.clear_all_signals)
        
        # Exportación
        self.export_csv_btn.clicked.connect(self.export_to_csv)
        self.export_plot_btn.clicked.connect(self.export_plot)
    
    def browse_blf_directory(self):
        """
        Abre diálogo para seleccionar directorio de archivos BLF.
        """
        directory = QFileDialog.getExistingDirectory(self, "Seleccionar Directorio BLF")
        if directory:
            self.blf_path_edit.setText(directory)
    
    def browse_dbc_file(self):
        """
        Abre diálogo para seleccionar archivo DBC.
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "Seleccionar Archivo DBC", "", "DBC files (*.dbc)")
        if file_path:
            self.dbc_path_edit.setText(file_path)
    
    def process_files(self):
        """
        Inicia el procesamiento de archivos BLF.
        """
        blf_directory = self.blf_path_edit.text().strip()
        dbc_path = self.dbc_path_edit.text().strip() or None
        
        if not blf_directory:
            QMessageBox.warning(self, "Error", "Debe seleccionar un directorio de archivos BLF")
            return
        
        if not os.path.exists(blf_directory):
            QMessageBox.warning(self, "Error", "El directorio BLF no existe")
            return
        
        if dbc_path and not os.path.exists(dbc_path):
            QMessageBox.warning(self, "Error", "El archivo DBC no existe")
            return
        
        # Deshabilitar controles durante procesamiento
        self.process_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Iniciar procesamiento en hilo separado
        self.processing_thread = ProcessingThread(blf_directory, dbc_path)
        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.processing_finished.connect(self.on_processing_finished)
        self.processing_thread.error_occurred.connect(self.on_processing_error)
        self.processing_thread.start()
    
    def update_progress(self, value, message):
        """
        Actualiza la barra de progreso.
        """
        self.progress_bar.setValue(value)
        self.statusBar().showMessage(message)
    
    def on_processing_finished(self, decoded_df):
        """
        Maneja la finalización del procesamiento.
        """
        self.current_data = decoded_df
        self.processor = self.processing_thread.processor
        
        # Re-habilitar controles
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("Procesamiento completado exitosamente")
        
        # Actualizar interfaz con datos procesados
        self.update_interface_with_data()
        
        # Mostrar mensaje de éxito
        QMessageBox.information(self, "Éxito", f"Procesamiento completado\n{len(decoded_df)} registros procesados")
    
    def on_processing_error(self, error_message):
        """
        Maneja errores durante el procesamiento.
        """
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("Error en procesamiento")
        
        QMessageBox.critical(self, "Error", f"Error durante el procesamiento:\n{error_message}")
    
    def update_interface_with_data(self):
        """
        Actualiza la interfaz con los datos procesados.
        """
        if self.current_data.empty:
            return
        
        # Actualizar combo de mensajes
        messages = self.processor.get_available_messages()
        self.message_combo.clear()
        self.message_combo.addItem("Todos los mensajes")
        self.message_combo.addItems(messages)
        
        # Actualizar lista de señales
        self.update_signals_list()
        
        # Actualizar estadísticas
        self.update_statistics()
        
        # Actualizar tabla de datos
        self.table_widget.update_table(self.current_data.head(1000))
    
    def update_signals_list(self):
        """
        Actualiza la lista de señales disponibles según el mensaje seleccionado.
        """
        self.signals_list.clear()
        
        if self.current_data.empty:
            return
        
        message_name = self.message_combo.currentText()
        
        if message_name == "Todos los mensajes":
            signals = self.processor.get_available_signals()
        else:
            signals = self.processor.get_available_signals(message_name)
        
        self.signals_list.addItems(signals)
    
    def add_selected_signals(self):
        """
        Agrega las señales seleccionadas al gráfico.
        """
        selected_items = self.signals_list.selectedItems()
        message_name = self.message_combo.currentText()
        
        for item in selected_items:
            signal_name = item.text()
            
            if signal_name in self.selected_signals:
                continue
            
            # Obtener datos de la señal
            if message_name == "Todos los mensajes":
                signal_data = self.processor.get_signal_data(signal_name=signal_name)
            else:
                signal_data = self.processor.get_signal_data(message_name, signal_name)
            
            if not signal_data.empty:
                # Agregar al gráfico
                self.plot_widget.plot_signal(signal_data, signal_name)
                self.selected_signals.add(signal_name)
        
        self.update_statistics()
    
    def remove_selected_signals(self):
        """
        Remueve las señales seleccionadas del gráfico.
        """
        selected_items = self.signals_list.selectedItems()
        
        for item in selected_items:
            signal_name = item.text()
            if signal_name in self.selected_signals:
                self.plot_widget.remove_signal(signal_name)
                self.selected_signals.discard(signal_name)
        
        self.update_statistics()
    
    def clear_all_signals(self):
        """
        Limpia todas las señales del gráfico.
        """
        self.plot_widget.clear_plots()
        self.selected_signals.clear()
        self.update_statistics()
    
    def update_statistics(self):
        """
        Actualiza las estadísticas mostradas.
        """
        if self.current_data.empty:
            self.stats_text.clear()
            return
        
        stats = []
        stats.append(f"Total de registros: {len(self.current_data):,}")
        stats.append(f"Mensajes únicos: {self.current_data['message_name'].nunique()}")
        stats.append(f"Señales únicas: {self.current_data['signal_name'].nunique()}")
        stats.append(f"Señales en gráfico: {len(self.selected_signals)}")
        
        if not self.current_data.empty:
            time_range = self.current_data['timestamp'].max() - self.current_data['timestamp'].min()
            stats.append(f"Duración: {time_range:.2f} segundos")
        
        self.stats_text.setPlainText('\n'.join(stats))
    
    def export_to_csv(self):
        """
        Exporta los datos actuales a CSV.
        """
        if self.current_data.empty:
            QMessageBox.warning(self, "Error", "No hay datos para exportar")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(self, "Exportar a CSV", "", "CSV files (*.csv)")
        if file_path:
            try:
                self.current_data.to_csv(file_path, index=False)
                QMessageBox.information(self, "Éxito", f"Datos exportados a {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error exportando datos: {str(e)}")
    
    def export_plot(self):
        """
        Exporta el gráfico actual como imagen.
        """
        if not self.selected_signals:
            QMessageBox.warning(self, "Error", "No hay señales en el gráfico para exportar")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(self, "Exportar Gráfico", "", "PNG files (*.png)")
        if file_path:
            try:
                exporter = pg.exporters.ImageExporter(self.plot_widget.plot_widget.plotItem)
                exporter.export(file_path)
                QMessageBox.information(self, "Éxito", f"Gráfico exportado a {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error exportando gráfico: {str(e)}")


def main():
    """
    Función principal para ejecutar la aplicación.
    """
    app = QApplication(sys.argv)
    
    # Configurar estilo
    app.setStyle('Fusion')
    
    # Crear y mostrar ventana principal
    window = CANVisualizerGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()