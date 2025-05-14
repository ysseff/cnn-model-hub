import sys
import os
import tensorflow as tf
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QProgressBar, QFileDialog, QTabWidget, QListWidget, QListWidgetItem,
    QDialog, QFormLayout, QLineEdit, QComboBox, QMessageBox, QRadioButton
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# ------------------ Layer Dialogs ------------------

class AddLayerDialog(QDialog):
    def __init__(self, image_type, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Layer")
        self.image_type = image_type
        self.selected_layer_info = None

        layout = QVBoxLayout()
        form_layout = QFormLayout()

        self.layer_type_combo = QComboBox()
        self.layer_type_combo.addItems([
            "Input", "Conv2D", "BatchNormalization", "MaxPooling2D",
            "Dropout", "Flatten", "Dense"
        ])
        self.layer_type_combo.currentTextChanged.connect(self.update_form_fields)
        form_layout.addRow("Layer Type:", self.layer_type_combo)

        self.param_fields = {}
        self.param_group = QWidget()
        self.param_form = QFormLayout()
        self.param_group.setLayout(self.param_form)
        layout.addLayout(form_layout)
        layout.addWidget(self.param_group)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self.accept_layer)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        self.setLayout(layout)
        self.update_form_fields()

    def update_form_fields(self):
        while self.param_form.rowCount() > 0:
            self.param_form.removeRow(0)
        self.param_fields.clear()

        layer_type = self.layer_type_combo.currentText()
        if layer_type == "Input":
            width_field = QLineEdit("128")
            width_field.setDisabled(True)
            height_field = QLineEdit("128")
            height_field.setDisabled(True)
            self.param_fields["Width"] = width_field
            self.param_fields["Height"] = height_field

            channels = "3" if self.image_type == "RGB" else "1"
            self.param_fields["Channels"] = QLabel(channels)

            self.param_form.addRow("Width:", width_field)
            self.param_form.addRow("Height:", height_field)
            self.param_form.addRow("Channels:", self.param_fields["Channels"])
        elif layer_type == "Conv2D":
            self.add_param("Filters", "32")
            self.add_param("Kernel", "3")
            self.add_param("Padding", "same")
            self.add_param("Activation", "relu")
        elif layer_type == "MaxPooling2D":
            self.add_param("Pool Size", "2")
        elif layer_type == "Dropout":
            self.add_param("Rate", "0.25")
        elif layer_type == "Dense":
            self.add_param("Units", "128")
            self.add_param("Activation", "relu")

    def add_param(self, name, default):
        field = QLineEdit(default)
        self.param_fields[name] = field
        self.param_form.addRow(f"{name}:", field)

    def accept_layer(self):
        layer_type = self.layer_type_combo.currentText()
        params = {k: v.text() if isinstance(v, QLineEdit) else v.text() for k, v in self.param_fields.items()}
        self.selected_layer_info = (layer_type, params)
        self.accept()

class CreateModelDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create New Model")
        self.setMinimumSize(600, 700)

        main_layout = QVBoxLayout()

        image_type_group = QGroupBox("Image Type")
        image_type_layout = QHBoxLayout()
        self.rgb_radio = QRadioButton("RGB")
        self.rgb_radio.setChecked(True)
        self.gray_radio = QRadioButton("Grayscale")
        image_type_layout.addWidget(self.rgb_radio)
        image_type_layout.addWidget(self.gray_radio)
        image_type_group.setLayout(image_type_layout)
        main_layout.addWidget(image_type_group)

        layer_group = QGroupBox("Layer Builder")
        layer_layout = QVBoxLayout()

        self.layer_list = QListWidget()
        layer_layout.addWidget(self.layer_list)

        btn_row = QHBoxLayout()
        add_layer_btn = QPushButton("Add Layer")
        add_layer_btn.clicked.connect(self.add_layer)
        default_model_btn = QPushButton("Add Default Model")
        default_model_btn.clicked.connect(self.add_default_model)
        btn_row.addWidget(add_layer_btn)
        btn_row.addWidget(default_model_btn)
        layer_layout.addLayout(btn_row)
        remove_layer_btn = QPushButton("Remove Selected Layer")
        remove_layer_btn.clicked.connect(self.remove_selected_layer)
        edit_layer_btn = QPushButton("Edit Selected Layer")
        edit_layer_btn.clicked.connect(self.edit_selected_layer)

        btn_row.addWidget(add_layer_btn)
        btn_row.addWidget(default_model_btn)
        btn_row.addWidget(remove_layer_btn)
        btn_row.addWidget(edit_layer_btn)

        layer_group.setLayout(layer_layout)
        main_layout.addWidget(layer_group)

        create_model_btn = QPushButton("Create Model")
        create_model_btn.clicked.connect(self.create_model)
        main_layout.addWidget(create_model_btn)

        self.setLayout(main_layout)

    def get_image_type(self):
        return "RGB" if self.rgb_radio.isChecked() else "Grayscale"

    def add_layer(self):
        dialog = AddLayerDialog(self.get_image_type(), self)
        if dialog.exec_() == QDialog.Accepted:
            layer_type, params = dialog.selected_layer_info
            self.insert_layer(layer_type, params)

    def insert_layer(self, layer_type, params):
        desc = f"{layer_type}: " + ', '.join(f"{k}={v}" for k, v in params.items())
        item = QListWidgetItem(desc)
        self.layer_list.addItem(item)

    def add_default_model(self):
        channels = "3" if self.get_image_type() == "RGB" else "1"

        default_model = [
            ("Input", {"Width": "128", "Height": "128", "Channels": channels}),
            ("Conv2D", {"Filters": "32", "Kernel": "3", "Padding": "same", "Activation": "relu"}),
            ("BatchNormalization", {}),
            ("MaxPooling2D", {"Pool Size": "2"}),
            ("Dropout", {"Rate": "0.25"}),
            ("Conv2D", {"Filters": "64", "Kernel": "3", "Padding": "same", "Activation": "relu"}),
            ("BatchNormalization", {}),
            ("MaxPooling2D", {"Pool Size": "2"}),
            ("Dropout", {"Rate": "0.25"}),
            ("Conv2D", {"Filters": "128", "Kernel": "3", "Padding": "same", "Activation": "relu"}),
            ("BatchNormalization", {}),
            ("MaxPooling2D", {"Pool Size": "2"}),
            ("Dropout", {"Rate": "0.25"}),
            ("Flatten", {}),
            ("Dense", {"Units": "128", "Activation": "relu"}),
            ("Dropout", {"Rate": "0.5"}),
            ("Dense", {"Units": "64", "Activation": "relu"})
        ]

        for layer_type, params in default_model:
            self.insert_layer(layer_type, params)

    def remove_selected_layer(self):
        selected_items = self.layer_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select a layer to remove.")
            return
        for item in selected_items:
            row = self.layer_list.row(item)
            self.layer_list.takeItem(row)

    def edit_selected_layer(self):
        selected_items = self.layer_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select a layer to edit.")
            return

        item = selected_items[0]
        text = item.text()
        layer_type, params_str = text.split(":", 1)
        params = {}
        for p in params_str.strip().split(','):
            if '=' in p:
                key, value = p.strip().split('=')
                params[key.strip()] = value.strip()

        # Open AddLayerDialog pre-filled with current params
        dialog = AddLayerDialog(self.get_image_type(), self)
        dialog.layer_type_combo.setCurrentText(layer_type.strip())
        dialog.update_form_fields()

        # Pre-fill param fields with existing values
        for key, field in dialog.param_fields.items():
            if key in params and isinstance(field, QLineEdit):
                field.setText(params[key])

        if dialog.exec_() == QDialog.Accepted:
            new_layer_type, new_params = dialog.selected_layer_info
            new_desc = f"{new_layer_type}: " + ', '.join(f"{k}={v}" for k, v in new_params.items())
            item.setText(new_desc)

    def create_model(self):
        layers = [self.layer_list.item(i).text() for i in range(self.layer_list.count())]
        QMessageBox.information(self, "Model Created", f"Model with {len(layers)} layers created!")
        self.accept()

# ------------------ Main App ------------------

class CNNModelHub(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CNN Model Hub")
        self.setGeometry(100, 100, 1200, 600)

        self.training_data_loaded = False
        self.validation_data_loaded = False
        self.model = None

        self.class_labels = []
        self.num_classes = 0

        self.datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
        self.test_datagen = ImageDataGenerator(rescale=1. / 255)

        self.image_display_label = QLabel("No image loaded")
        self.image_display_label.setAlignment(Qt.AlignCenter)
        self.confidence_label = QLabel("Confidence: N/A")
        self.confidence_label.setAlignment(Qt.AlignCenter)

        main_layout = QHBoxLayout()
        self.central_widget = QWidget()
        self.central_widget.setLayout(main_layout)
        self.setCentralWidget(self.central_widget)

        main_layout.addWidget(self.data_management_panel())
        main_layout.addWidget(self.training_validation_panel())
        main_layout.addWidget(self.visualization_panel())

    def get_image_type(self):
        return 'rgb' if self.rgb_radio.isChecked() else 'grayscale'

    # ------------------ Data Management ------------------

    def data_management_panel(self):
        panel = QGroupBox("Data Management")
        layout = QVBoxLayout()

        image_type_group = QGroupBox("Image Type")
        image_type_layout = QHBoxLayout()
        self.rgb_radio = QRadioButton("RGB")
        self.rgb_radio.setChecked(True)
        self.gray_radio = QRadioButton("Grayscale")
        image_type_layout.addWidget(self.rgb_radio)
        image_type_layout.addWidget(self.gray_radio)
        image_type_group.setLayout(image_type_layout)
        layout.addWidget(image_type_group)

        self.train_data_label = QLabel("No data loaded")
        upload_train_btn = QPushButton("Upload Training Data")
        upload_train_btn.clicked.connect(self.upload_training_data)
        layout.addWidget(upload_train_btn)
        layout.addWidget(self.train_data_label)

        self.val_data_label = QLabel("No data loaded")
        upload_val_btn = QPushButton("Upload Validation Data")
        upload_val_btn.clicked.connect(self.upload_validation_data)
        layout.addWidget(upload_val_btn)
        layout.addWidget(self.val_data_label)

        self.classes_info_label = QLabel("Classes: N/A")
        layout.addWidget(self.classes_info_label)

        create_model_btn = QPushButton("Create New Model")
        create_model_btn.clicked.connect(self.open_create_model_dialog)
        load_model_btn = QPushButton("Load Pretrained Model (.keras)")
        load_model_btn.clicked.connect(self.load_pretrained_model)
        layout.addStretch()
        layout.addWidget(create_model_btn)
        layout.addWidget(load_model_btn)

        panel.setLayout(layout)
        return panel

    def upload_training_data(self):
        dir_name = QFileDialog.getExistingDirectory(self, "Select Training Data Folder")
        if dir_name:
            self.train_data_dir = dir_name
            self.train_data_label.setText(os.path.basename(dir_name))
            self.training_data_loaded = True

    def upload_validation_data(self):
        dir_name = QFileDialog.getExistingDirectory(self, "Select Validation Data Folder")
        if dir_name:
            self.val_data_dir = dir_name
            self.val_data_label.setText(os.path.basename(dir_name))
            self.validation_data_loaded = True
            subfolders = sorted([name for name in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name, name))])
            self.class_labels = subfolders
            self.num_classes = len(subfolders)
            self.classes_info_label.setText(f"Classes Found: {self.num_classes} | Labels: {self.class_labels}")

    # ------------------ Model Management ------------------

    def open_create_model_dialog(self):
        if not (self.training_data_loaded and self.validation_data_loaded):
            QMessageBox.warning(self, "Missing Data", "Please load both Training and Validation data first.")
            return
        dialog = CreateModelDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            self.build_model(dialog.layer_list)

    def build_model(self, layer_list):
        layers_info = []
        for i in range(layer_list.count()):
            text = layer_list.item(i).text()
            layer_type, params_str = text.split(":", 1)
            params = {}
            for p in params_str.strip().split(','):
                if '=' in p:
                    key, value = p.strip().split('=')
                    params[key.strip()] = value.strip()
            layers_info.append((layer_type.strip(), params))

        model = tf.keras.Sequential()
        for layer_type, params in layers_info:
            if layer_type == "Input":
                model.add(tf.keras.layers.InputLayer(
                    input_shape=(int(params["Height"]), int(params["Width"]), int(params["Channels"]))))
            elif layer_type == "Conv2D":
                model.add(tf.keras.layers.Conv2D(int(params["Filters"]), (int(params["Kernel"]), int(params["Kernel"])),
                                                 padding=params["Padding"], activation=params["Activation"]))
            elif layer_type == "BatchNormalization":
                model.add(tf.keras.layers.BatchNormalization())
            elif layer_type == "MaxPooling2D":
                model.add(tf.keras.layers.MaxPooling2D(pool_size=(int(params["Pool Size"]), int(params["Pool Size"]))))
            elif layer_type == "Dropout":
                model.add(tf.keras.layers.Dropout(float(params["Rate"])))
            elif layer_type == "Flatten":
                model.add(tf.keras.layers.Flatten())
            elif layer_type == "Dense":
                model.add(tf.keras.layers.Dense(int(params["Units"]), activation=params["Activation"]))

        if self.num_classes == 2:
            model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
            loss_fn = 'binary_crossentropy'
        else:
            model.add(tf.keras.layers.Dense(self.num_classes, activation="softmax"))
            loss_fn = 'categorical_crossentropy'

        model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
        self.model = model
        QMessageBox.information(self, "Model Ready", "New model compiled and ready.")

    def load_pretrained_model(self):
        if not self.validation_data_loaded:
            QMessageBox.warning(self, "Missing Validation Data", "Load validation data first.")
            return
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Pretrained Model", "", "Keras Model (*.keras)")
        if file_name:
            self.model = tf.keras.models.load_model(file_name)
            QMessageBox.information(self, "Model Loaded", f"Loaded model: {os.path.basename(file_name)}")

    # ------------------ Training, Validation, Classification ------------------

    def train_model(self):
        if not self.model:
            QMessageBox.warning(self, "No Model", "Load or create a model first.")
            return

        output_shape = self.model.output_shape[-1]
        class_mode = 'binary' if output_shape == 1 else 'categorical'
        loss_fn = 'binary_crossentropy' if output_shape == 1 else 'categorical_crossentropy'

        color_mode = self.get_image_type()

        train_gen = self.datagen.flow_from_directory(
            self.train_data_dir,
            target_size=(128, 128),
            batch_size=32,
            class_mode=class_mode,
            color_mode=color_mode
        )

        val_gen = self.test_datagen.flow_from_directory(
            self.val_data_dir,
            target_size=(128, 128),
            batch_size=32,
            class_mode=class_mode,
            color_mode=color_mode
        )

        self.model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
        try:
            epochs = int(self.epochs_input.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Epochs", "Please enter a valid number of epochs.")
            return

        checkpoint_callback = ModelCheckpoint(
            'best.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1,
            save_weights_only=False
        )

        history = self.model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=[checkpoint_callback])
        self.model.save('best.keras')
        # Visualization update
        history_dict = history.history

        # Accuracy plot
        self.accuracy_ax.clear()
        self.accuracy_ax.plot(history_dict['accuracy'], label='Train Acc')
        self.accuracy_ax.plot(history_dict['val_accuracy'], label='Val Acc')
        self.accuracy_ax.set_title('Accuracy')
        self.accuracy_ax.legend()
        self.accuracy_canvas.draw()

        # Loss plot
        self.loss_ax.clear()
        self.loss_ax.plot(history_dict['loss'], label='Train Loss')
        self.loss_ax.plot(history_dict['val_loss'], label='Val Loss')
        self.loss_ax.set_title('Loss')
        self.loss_ax.legend()
        self.loss_canvas.draw()
        QMessageBox.information(self, "Training Complete", "Model trained and saved to 'trained_model.keras'.")

    def validate_model(self):
        if not self.model:
            QMessageBox.warning(self, "No Model", "Load or create a model first.")
            return

        output_shape = self.model.output_shape[-1]
        class_mode = 'binary' if output_shape == 1 else 'categorical'
        loss_fn = 'binary_crossentropy' if output_shape == 1 else 'categorical_crossentropy'
        color_mode = self.get_image_type()

        val_gen = self.test_datagen.flow_from_directory(
            self.val_data_dir,
            target_size=(128, 128),
            batch_size=32,
            class_mode=class_mode,
            color_mode=color_mode
        )

        self.model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
        loss, acc = self.model.evaluate(val_gen)
        QMessageBox.information(self, "Validation Results", f"Validation Accuracy: {acc:.2%}")

    def classify_image(self):
        if not self.model:
            QMessageBox.warning(self, "No Model", "Load or create a model first.")
            return

        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_name:
            color_mode = self.get_image_type()

            # Load image with correct color_mode
            img = tf.keras.preprocessing.image.load_img(file_name, target_size=(128, 128), color_mode=color_mode)
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            img_array = tf.expand_dims(img_array, axis=0)

            prediction = self.model.predict(img_array)
            if self.model.output_shape[-1] == 1:
                raw_confidence = prediction[0][0]
                class_idx = int(raw_confidence > 0.5)
                confidence = raw_confidence if class_idx == 1 else 1 - raw_confidence
            else:
                confidence = tf.reduce_max(prediction[0]).numpy()
                class_idx = tf.argmax(prediction[0]).numpy()

            predicted_class = self.class_labels[class_idx]

            # Display image preview
            pixmap = QPixmap(file_name).scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_display_label.setPixmap(pixmap)

            # Update confidence label
            self.confidence_label.setText(f"Class: {predicted_class} | Confidence: {confidence:.2%}")

    # ------------------ Panels ------------------

    def training_validation_panel(self):
        panel = QGroupBox("Training & Validation")
        layout = QVBoxLayout()

        layout.addWidget(self.image_display_label)
        layout.addWidget(self.confidence_label)

        epochs_group = QGroupBox("Training Parameters")
        epochs_layout = QHBoxLayout()
        self.epochs_input = QLineEdit("5")  # Default to 5
        epochs_layout.addWidget(QLabel("Epochs:"))
        epochs_layout.addWidget(self.epochs_input)
        epochs_group.setLayout(epochs_layout)
        layout.addWidget(epochs_group)

        train_btn = QPushButton("Train Model")
        train_btn.clicked.connect(self.train_model)
        validate_btn = QPushButton("Validate Model")
        validate_btn.clicked.connect(self.validate_model)
        classify_btn = QPushButton("Test Image Classification")
        classify_btn.clicked.connect(self.classify_image)

        layout.addWidget(train_btn)
        layout.addWidget(validate_btn)
        layout.addWidget(classify_btn)

        panel.setLayout(layout)
        return panel

    def visualization_panel(self):
        panel = QGroupBox("Visualization")
        layout = QVBoxLayout()
        self.tabs = QTabWidget()

        # Accuracy Tab
        self.accuracy_fig = Figure()
        self.accuracy_canvas = FigureCanvas(self.accuracy_fig)
        self.accuracy_ax = self.accuracy_fig.add_subplot(111)
        acc_tab = QWidget()
        acc_layout = QVBoxLayout()
        acc_layout.addWidget(self.accuracy_canvas)
        acc_tab.setLayout(acc_layout)
        self.tabs.addTab(acc_tab, "Accuracy")

        # Loss Tab
        self.loss_fig = Figure()
        self.loss_canvas = FigureCanvas(self.loss_fig)
        self.loss_ax = self.loss_fig.add_subplot(111)
        loss_tab = QWidget()
        loss_layout = QVBoxLayout()
        loss_layout.addWidget(self.loss_canvas)
        loss_tab.setLayout(loss_layout)
        self.tabs.addTab(loss_tab, "Loss")

        layout.addWidget(self.tabs)
        panel.setLayout(layout)
        return panel

# ------------------ Run ------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CNNModelHub()
    window.show()
    sys.exit(app.exec_())