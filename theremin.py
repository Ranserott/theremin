#!/usr/bin/env python3
"""
Camera Theremin - Un theremin virtual controlado por cámara
Usa MediaPipe para detectar la mano y OpenCV para procesamiento de video
Genera sonido en tiempo real usando numpy y sounddevice
"""

import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import threading
import time
from collections import deque
import math


class NoteDetector:
    """Clase para detectar la nota musical más cercana a una frecuencia"""

    def __init__(self):
        # Notas musicales con sus frecuencias (temperamento igual)
        self.notes = [
            ('C3', 130.81), ('C#3', 138.59), ('D3', 146.83), ('D#3', 155.56),
            ('E3', 164.81), ('F3', 174.61), ('F#3', 185.00), ('G3', 196.00),
            ('G#3', 207.65), ('A3', 220.00), ('A#3', 233.08), ('B3', 246.94),
            ('C4', 261.63), ('C#4', 277.18), ('D4', 293.66), ('D#4', 311.13),
            ('E4', 329.63), ('F4', 349.23), ('F#4', 369.99), ('G4', 392.00),
            ('G#4', 415.30), ('A4', 440.00), ('A#4', 466.16), ('B4', 493.88),
            ('C5', 523.25), ('C#5', 554.37), ('D5', 587.33), ('D#5', 622.25),
            ('E5', 659.25), ('F5', 698.46), ('F#5', 739.99), ('G5', 783.99),
            ('G#5', 830.61), ('A5', 880.00), ('A#5', 932.33), ('B5', 987.77),
            ('C6', 1046.50), ('C#6', 1108.73), ('D6', 1174.66), ('D#6', 1244.51),
            ('E6', 1318.51), ('F6', 1396.91), ('F#6', 1479.98), ('G6', 1567.98),
            ('G#6', 1661.22), ('A6', 1760.00), ('A#6', 1864.66), ('B6', 1975.53)
        ]

    def get_closest_note(self, frequency):
        """Obtiene la nota musical más cercana a la frecuencia dada"""
        if frequency < 150 or frequency > 2000:
            return None, 0

        # Encontrar la nota más cercana
        closest_note = None
        min_diff = float('inf')

        for note_name, note_freq in self.notes:
            diff = abs(frequency - note_freq)
            if diff < min_diff:
                min_diff = diff
                closest_note = (note_name, note_freq)

        # Calcular cuántos cents de diferencia
        if closest_note:
            note_name, note_freq = closest_note
            cents = 1200 * math.log2(frequency / note_freq)
            return closest_note, round(cents, 1)

        return None, 0


class HandDetector:
    """Clase para detectar y seguir ambas manos usando MediaPipe"""

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Detectar hasta 2 manos
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def find_hands(self, frame):
        """Detecta ambas manos y devuelve coordenadas separadas para tono y volumen"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        right_hand = None  # Controla el tono (pitch)
        left_hand = None   # Controla el volumen

        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                # Determinar si es mano derecha o izquierda
                hand_label = handedness.classification[0].label
                confidence = handedness.classification[0].score

                # Dibujar la mano y puntos clave
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                h, w, _ = frame.shape

                # Usar diferentes puntos de control según la mano
                if hand_label == 'Right':
                    # Mano derecha - control de tono usando dedo índice (landmark 8)
                    index_finger_x = hand_landmarks.landmark[8].x
                    index_finger_y = hand_landmarks.landmark[8].y
                    cx, cy = int(index_finger_x * w), int(index_finger_y * h)
                    
                    # Resaltar dedo índice para control de tono (color amarillo)
                    cv2.circle(frame, (cx, cy), 15, (0, 255, 255), -1)
                    cv2.circle(frame, (cx, cy), 18, (0, 200, 200), 3)
                    cv2.putText(frame, "TONO", (cx - 40, cy - 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    right_hand = (index_finger_x, index_finger_y, (cx, cy))
                else:
                    # Mano izquierda - control de volumen usando palma (landmark 9)
                    palm_x = hand_landmarks.landmark[9].x
                    palm_y = hand_landmarks.landmark[9].y
                    cx, cy = int(palm_x * w), int(palm_y * h)
                    
                    # Resaltar palma para control de volumen (color cian)
                    cv2.circle(frame, (cx, cy), 12, (255, 255, 0), -1)
                    cv2.putText(frame, "VOL", (cx - 20, cy - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    left_hand = (palm_x, palm_y, (cx, cy))

        return right_hand, left_hand


class AudioGenerator:
    """Clase para generar audio ultra simple y limpio - solo ondas sinusoidales puras"""

    def __init__(self, sample_rate=44100, buffer_size=128):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.frequency = 440.0  # Frecuencia inicial (A4)
        self.volume = 0.0      # Volumen inicial (silencio)
        self.phase = 0.0       # Fase para generar onda continua
        self.is_playing = False
        
        # Solo un tipo de onda: sinusoidal pura
        self.current_wave = 'pure_sine'
        
        # Filtros mínimos
        self.freq_filter = deque(maxlen=8)
        self.vol_filter = deque(maxlen=6)

    def update_parameters(self, frequency, volume):
        """Actualiza la frecuencia y volumen con filtrado mínimo"""
        # Filtrado muy simple
        self.freq_filter.append(frequency)
        self.vol_filter.append(volume)
        
        # Promedio simple
        self.frequency = np.mean(self.freq_filter) if self.freq_filter else frequency
        self.volume = np.mean(self.vol_filter) if self.vol_filter else volume

    def generate_callback(self, outdata, frames, time_info, status):
        """Callback ultra simple - solo onda sinusoidal pura"""
        
        # Silencio si no hay volumen
        if self.volume < 0.01:
            outdata.fill(0)
            return
        
        # Generar onda sinusoidal pura y simple
        phase_increment = 2 * np.pi * self.frequency / self.sample_rate
        
        # Crear array de fases
        phases = self.phase + np.arange(frames) * phase_increment
        
        # Generar señal sinusoidal pura
        signal = np.sin(phases) * self.volume * 0.3  # Volumen reducido para evitar distorsión
        
        # Actualizar fase para continuidad
        self.phase = phases[-1] % (2 * np.pi)
        
        # Asignar directamente sin procesamiento adicional
        outdata[:, 0] = signal.astype(np.float32)

    def start(self):
        """Inicia el stream de audio"""
        if not self.is_playing:
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self.generate_callback,
                blocksize=self.buffer_size
            )
            self.stream.start()
            self.is_playing = True

    def stop(self):
        """Detiene el stream de audio"""
        if self.is_playing:
            self.stream.stop()
            self.stream.close()
            self.is_playing = False


class CameraTheremin:
    """Clase principal del theremin controlado por cámara dual"""

    def __init__(self):
        self.hand_detector = HandDetector()
        self.audio_generator = AudioGenerator()
        self.note_detector = NoteDetector()
        self.cap = None

        # Parámetros de mapeo
        self.min_freq = 200.0   # Frecuencia mínima (Hz)
        self.max_freq = 2000.0  # Frecuencia máxima (Hz)
        self.min_volume = 0.0   # Volumen mínimo
        self.max_volume = 0.8   # Volumen máximo (para evitar distorsión)

        # Suavizado de parámetros para reducir fluctuaciones
        self.freq_history = deque(maxlen=5)
        self.vol_history = deque(maxlen=5)

        # Últimos valores conocidos para cuando falta una mano
        self.last_frequency = 440.0
        self.last_volume = 0.0

    def map_frequency(self, hand_x):
        """Mapea la coordenada X de la mano derecha a frecuencia"""
        # X: 0 (izquierda) -> max_freq (agudo), X: 1 (derecha) -> min_freq (grave)
        # Como en el theremin real: izquierda = tonos altos, derecha = tonos bajos
        frequency = self.max_freq - (hand_x * (self.max_freq - self.min_freq))
        return frequency

    def map_volume(self, hand_y):
        """Mapea la coordenada Y de la mano izquierda a volumen"""
        # Y: 0 (arriba) -> max_volume, Y: 1 (abajo) -> min_volume
        # Como en el theremin real: mano cerca = volumen alto, mano lejos = volumen bajo
        volume = self.max_volume - (hand_y * self.max_volume)
        return volume

    def process_dual_control(self, right_hand, left_hand):
        """Procesa el control dual de ambas manos como un theremin real"""
        frequency = self.last_frequency
        volume = self.last_volume
        right_detected = right_hand is not None
        left_detected = left_hand is not None

        # Procesar mano derecha (tono)
        if right_detected:
            right_x, right_y, _ = right_hand
            frequency = self.map_frequency(right_x)
            self.last_frequency = frequency

        # Procesar mano izquierda (volumen)
        if left_detected:
            left_x, left_y, _ = left_hand
            volume = self.map_volume(left_y)
            self.last_volume = volume

        # Lógica del theremin real:
        # Si no hay mano izquierda -> sin sonido
        # Si hay mano izquierda pero no derecha -> mantener último tono
        # Si hay mano derecha pero no izquierda -> sin sonido
        # Si ambas manos -> control completo

        if not left_detected:
            # Sin mano izquierda = silencio (como el theremin real)
            volume = 0.0

        return frequency, volume, right_detected, left_detected

    def smooth_parameters(self, freq, vol):
        """Suaviza los parámetros para reducir fluctuaciones"""
        self.freq_history.append(freq)
        self.vol_history.append(vol)

        # Usar mediana para reducir valores atípicos
        if len(self.freq_history) > 0:
            smooth_freq = np.median(self.freq_history)
            smooth_vol = np.median(self.vol_history)
        else:
            smooth_freq = freq
            smooth_vol = vol

        return smooth_freq, smooth_vol

    def draw_info(self, frame, frequency, volume, right_detected, left_detected):
        """Dibuja información compacta en pantalla con notas musicales"""
        h, w = frame.shape[:2]

        # Fondo semitransparente compacto
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (320, 140), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 1, overlay, 0.6, 0)

        # Detectar nota musical
        note_info = self.note_detector.get_closest_note(frequency)
        if note_info[0]:
            note_name, note_freq = note_info[0]
            cents_diff = note_info[1]
            note_text = f"♪ {note_name}"
            cents_text = f"({cents_diff:+.0f}¢)"
        else:
            note_text = "♪ --"
            cents_text = ""

        # Estados compactos de las manos
        right_color = (0, 255, 255) if right_detected else (0, 100, 100)
        left_color = (255, 255, 0) if left_detected else (100, 100, 0)

        # Estados de detección (texto más pequeño)
        cv2.putText(frame, f"DERECHA: {'✓' if right_detected else '✗'}", (12, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, right_color, 1)
        cv2.putText(frame, f"IZQUIERDA: {'✓' if left_detected else '✗'}", (12, 48),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, left_color, 1)

        # Nota musical grande y destacada
        cv2.putText(frame, note_text, (12, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 220, 100), 2)
        if cents_text:
            cv2.putText(frame, cents_text, (140, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Frecuencia y volumen (texto más pequeño)
        cv2.putText(frame, f"{frequency:.0f} Hz", (12, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"Vol: {volume:.0%}", (80, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Estado general (compacto)
        if left_detected and right_detected:
            status_text = "● ACTIVO"
            status_color = (0, 255, 0)
        elif left_detected or right_detected:
            status_text = "○ PARCIAL"
            status_color = (255, 200, 0)
        else:
            status_text = "× INACTIVO"
            status_color = (100, 100, 100)

        cv2.putText(frame, status_text, (12, 125),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)

        # Tipo de sonido compacto
        wave_names = {
            'harmonic_sine': 'Sine Armónico',
            'warm_strings': 'Cuerdas Cálidas ⭐',
            'soft_bell': 'Campana Suave',
            'gentle_pad': 'Pad Suave'
        }
        current_wave_name = wave_names.get(self.audio_generator.current_wave, 'Desconocido')
        cv2.putText(frame, current_wave_name, (150, 125),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 150, 255), 1)

        # Indicadores visuales compactos
        # Barra de frecuencia
        freq_normalized = max(0, min(1, (frequency - self.min_freq) / (self.max_freq - self.min_freq)))
        freq_bar_width = int(freq_normalized * 120)
        cv2.rectangle(frame, (12, 145), (132, 155), (40, 40, 40), -1)
        cv2.rectangle(frame, (12, 145), (12 + freq_bar_width, 155), (0, 255, 255), -1)

        # Barra de volumen
        vol_bar_width = int(volume / self.max_volume * 120)
        cv2.rectangle(frame, (140, 145), (260, 155), (40, 40, 40), -1)
        cv2.rectangle(frame, (140, 145), (140 + vol_bar_width, 155), (255, 255, 0), -1)

        # Controles miniatura
        cv2.putText(frame, "Q: Salir", (270, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 200), 1)

        # Instrucciones ultra compactas
        cv2.putText(frame, "DERECHA (amarilla)", (270, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 200, 200), 1)
        cv2.putText(frame, "IZQUIERDA (cian)", (270, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.25, (200, 200, 0), 1)

        return frame

    def change_wave_type(self, wave_number):
        """Cambia el tipo de onda según el número"""
        if 1 <= wave_number <= len(self.audio_generator.wave_types):
            new_wave = self.audio_generator.wave_types[wave_number - 1]
            self.audio_generator.current_wave = new_wave
            print(f"🎵 Cambiado a sonido: {new_wave}")
            return True
        return False

    def run(self):
        """Ejecuta el theremin"""
        try:
            # Inicializar cámara
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            if not self.cap.isOpened():
                print("Error: No se pudo abrir la cámara")
                return

            print("🎵 Camera Theremin DUAL - Sonido Ultra Suave")
            print("📋 Controles:")
            print("  🖐️  Mano DERECHA (amarilla): TONO")
            print("     ← Izquierda: Tonos altos | → Derecha: Tonos bajos")
            print("  🖐️  Mano IZQUIERDA (cian): VOLUMEN")
            print("     ↑ Arriba: Volumen alto | ↓ Abajo: Volumen bajo")
            print("  🎹 Teclas 1-4: Sonidos armónicos")
            print("     1: Sine Armónico | 2: Cuerdas Cálidas ⭐")
            print("     3: Campana Suave | 4: Pad Suave")
            print("  ♪ Verás la nota musical en tiempo real")
            print("  🚪 Tecla 'q': Salir")
            print()

            # Iniciar audio
            self.audio_generator.start()

            # Variables para FPS
            last_time = time.time()
            frame_count = 0

            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: No se pudo leer el frame")
                    break

                # Voltear frame horizontalmente (efecto espejo)
                frame = cv2.flip(frame, 1)

                # Detectar ambas manos
                right_hand, left_hand = self.hand_detector.find_hands(frame)

                # Procesar control dual del theremin
                frequency, volume, right_detected, left_detected = self.process_dual_control(right_hand, left_hand)

                # Suavizar parámetros para reducir fluctuaciones
                frequency, volume = self.smooth_parameters(frequency, volume)

                # DEBUG: Mostrar valores para diagnóstico
                if right_detected and left_detected and volume > 0.01:
                    print(f"DEBUG: Freq={frequency:.1f}Hz, Vol={volume:.3f}, Wave={self.audio_generator.current_wave}")

                # Actualizar generador de audio
                self.audio_generator.update_parameters(frequency, volume)

                # Dibujar información en pantalla
                frame = self.draw_info(frame, frequency, volume, right_detected, left_detected)

                # Calcular y mostrar FPS
                frame_count += 1
                current_time = time.time()
                if current_time - last_time > 1.0:
                    fps = frame_count / (current_time - last_time)
                    cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 100, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    frame_count = 0
                    last_time = current_time

                # Mostrar frame
                cv2.imshow('Camera Theremin', frame)

                # Procesar teclas
                key = cv2.waitKey(1) & 0xFF

                # Salir con 'q'
                if key == ord('q'):
                    break

                # Cambiar tipo de sonido con números 1-4
                if key >= ord('1') and key <= ord('4'):
                    wave_number = key - ord('1') + 1
                    self.change_wave_type(wave_number)

        except KeyboardInterrupt:
            print("\nInterrupción del usuario")

        except Exception as e:
            print(f"Error: {e}")

        finally:
            # Limpiar recursos
            print("Cerrando...")
            if self.audio_generator.is_playing:
                self.audio_generator.stop()
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()


def main():
    """Función principal"""
    theremin = CameraTheremin()
    theremin.run()


if __name__ == "__main__":
    main()