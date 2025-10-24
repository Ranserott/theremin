# 🎵 Camera Theremin

Un theremin virtual controlado por cámara usando visión por computadora y procesamiento de audio en tiempo real.

## 📋 Descripción

Este proyecto simula un theremin musical utilizando la cámara web para detectar la posición de la mano en tiempo real mediante **MediaPipe**. La posición horizontal controla la frecuencia del tono (pitch) y la posición vertical controla el volumen, generando sonido continuo con **numpy** y **sounddevice**.

## ✨ Características

- 🖐️ Detección de mano en tiempo real con MediaPipe
- 🎵 Generación de sonido sinusoidal continuo sin interrupciones
- 📊 Visualización en vivo con indicadores de frecuencia y volumen
- 🔄 Suavizado de parámetros para reducir fluctuaciones
- 🔇 Silenciado automático al no detectar mano
- ⚡ Optimizado para baja latencia
- 🎯 Interfaz intuitiva con feedback visual

## 🛠️ Instalación Rápida

### Opción 1: Configuración automática (recomendada)
```bash
python setup.py
```

### Opción 2: Instalación manual
1. Asegúrate de tener Python 3.8+
2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## 🎮 Uso

Ejecutar el script principal:
```bash
python theremin.py
```

Para salir, presiona la tecla **'q'** en la ventana de la cámara.

## 📹 Controles

| Movimiento | Parámetro | Rango | Dirección |
|------------|-----------|-------|-----------|
| **Horizontal** (izquierda ↔ derecha) | **Frecuencia** | 200Hz - 2000Hz | Izquierda: Agudo ↔ Derecha: Grave |
| **Vertical** (arriba ↔ abajo) | **Volumen** | 0% - 80% | Arriba: Alto ↔ Abajo: Bajo |
| **Sin mano** | **Silencio** | - | Silencio automático |

## 🎯 Consejos de Uso

- Mantén tu mano bien iluminada para mejor detección
- Mueve la mano suavemente para evitar cambios bruscos
- La palma de la mano se usa como punto de control (landmark 9)
- El efecto espejo está activado para movimiento intuitivo

## ⚙️ Requisitos del Sistema

- **Python 3.8+**
- **Cámara web** (USB o integrada)
- **Sistema de audio** (speakers o auriculares)
- **Sistema Operativo**: Windows, macOS o Linux

## 🏗️ Estructura del Proyecto

```
theremi/
├── theremin.py       # Script principal del theremin
├── setup.py          # Script de configuración y verificación
├── requirements.txt  # Dependencias del proyecto
└── README.md        # Documentación
```

## 🔧 Arquitectura

El proyecto está organizado en tres clases principales:

- **`HandDetector`**: Detección y seguimiento de manos con MediaPipe
- **`AudioGenerator`**: Generación de audio en tiempo real con numpy/sounddevice
- **`CameraTheremin`**: Coordinación principal y visualización

## 🐛 Solución de Problemas

### Problemas comunes:

**❌ "No se pudo abrir la cámara"**
- Verifica que la cámara no esté en uso por otra aplicación
- Asegúrate de tener los drivers de cámara actualizados

**❌ "Error de audio"**
- Verifica que tengas speakers o auriculares conectados
- Ajusta el volumen del sistema

**❌ "No se detecta la mano"**
- Mejora la iluminación de la habitación
- Asegúrate de que tu mano esté visible en la cámara
- Evita fondos muy complejos

**❌ Dependencias faltantes**
```bash
# Reinstalar todo
pip uninstall opencv-python mediapipe numpy sounddevice -y
pip install -r requirements.txt
```

## 🎵 Características Técnicas

- **Sample rate**: 44.1 kHz
- **Buffer size**: 128 samples (baja latencia)
- **Formato de audio**: Float32, mono
- **Frame rate**: ~30 FPS (depende de la cámara)
- **Frecuencia**: Onda sinusoidal pura
- **Latencia**: <10ms (depende del hardware)

By Francisco Cerda Escobar