#!/usr/bin/env python3
"""
Script de configuración para Camera Theremin
Verifica dependencias y configura el sistema
"""

import subprocess
import sys
import os


def check_python_version():
    """Verifica la versión de Python"""
    if sys.version_info < (3, 8):
        print("❌ Error: Se requiere Python 3.8 o superior")
        print(f"   Versión actual: {sys.version}")
        return False
    else:
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        return True


def check_dependencies():
    """Verifica las dependencias instaladas"""
    dependencies = ['cv2', 'mediapipe', 'numpy', 'sounddevice']
    missing_deps = []

    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep} instalado")
        except ImportError:
            print(f"❌ {dep} no encontrado")
            missing_deps.append(dep)

    return missing_deps


def install_dependencies(missing_deps):
    """Instala las dependencias faltantes"""
    if not missing_deps:
        return True

    print("\n📦 Instalando dependencias faltantes...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_deps)
        print("✅ Dependencias instaladas correctamente")
        return True
    except subprocess.CalledProcessError:
        print("❌ Error al instalar dependencias")
        print("   Por favor, ejecuta: pip install -r requirements.txt")
        return False


def check_camera():
    """Verifica si hay cámara disponible"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Cámara detectada")
            cap.release()
            return True
        else:
            print("❌ No se pudo acceder a la cámara")
            return False
    except Exception as e:
        print(f"❌ Error al verificar cámara: {e}")
        return False


def check_audio():
    """Verifica si hay sistema de audio"""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        output_devices = [d for d in devices if d['max_output_channels'] > 0]

        if output_devices:
            print(f"✅ Sistema de audio detectado ({len(output_devices)} dispositivos)")
            return True
        else:
            print("❌ No se encontraron dispositivos de audio")
            return False
    except Exception as e:
        print(f"❌ Error al verificar audio: {e}")
        return False


def main():
    """Función principal de configuración"""
    print("🎵 Camera Theremin - Configuración\n")

    # Verificar versión de Python
    if not check_python_version():
        sys.exit(1)

    # Verificar dependencias
    missing_deps = check_dependencies()

    if missing_deps:
        print(f"\n📦 Dependencias faltantes: {', '.join(missing_deps)}")

        response = input("¿Desea instalarlas automáticamente? (y/n): ").lower().strip()
        if response in ['y', 'yes', 'sí', 'si']:
            if not install_dependencies(missing_deps):
                sys.exit(1)
        else:
            print("❌ Instale las dependencias manualmente:")
            print("   pip install -r requirements.txt")
            sys.exit(1)

    # Verificar hardware
    print("\n🔍 Verificando hardware...")
    camera_ok = check_camera()
    audio_ok = check_audio()

    # Resumen
    print("\n📋 Resumen de la configuración:")
    print(f"   Python: {'✅' if check_python_version() else '❌'}")
    print(f"   Dependencias: {'✅' if not missing_deps else '❌'}")
    print(f"   Cámara: {'✅' if camera_ok else '❌'}")
    print(f"   Audio: {'✅' if audio_ok else '❌'}")

    if camera_ok and audio_ok and not missing_deps:
        print("\n🎉 ¡Configuración completada!")
        print("\nPara iniciar el theremin, ejecuta:")
        print("   python theremin.py")
        print("\nPara salir, presiona 'q' en la ventana de la cámara")
    else:
        print("\n⚠️  Hay problemas que deben resolverse antes de usar el theremin")
        if not camera_ok:
            print("   - Verifica que tu cámara esté conectada y no esté en uso")
        if not audio_ok:
            print("   - Verifica que tengas speakers o auriculares conectados")

    return camera_ok and audio_ok and not missing_deps


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)