import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import keyboard  # Biblioteca para detectar pressionamento de teclas

# Inicializando o Mediapipe e a câmera
mp_face_mesh = mp.solutions.face_mesh
mp_face_mesh_instance = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
cap = cv2.VideoCapture(0)

# Obtém a resolução da tela
screen_w, screen_h = pyautogui.size()

# Variáveis para controle do mouse
current_mouse_x, current_mouse_y = pyautogui.position()
speed = 25  # Incremento de movimentação
smooth_factor = 0.2  # Fator de suavização
max_speed = 50  # Velocidade máxima
min_speed = 1   # Velocidade mínima

# Variáveis para detecção de piscadas
initial_left_eye_height = None
initial_right_eye_height = None
eyes_blinked = False

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    # Flip da imagem
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Processa a imagem para detectar landmarks
    results = mp_face_mesh_instance.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Pega as coordenadas do nariz
            nose = face_landmarks.landmark[1]  # Landmark do nariz

            # Obtenha a posição da cabeça
            h, w, _ = frame.shape
            head_x = int(nose.x * w)
            head_y = int(nose.y * h)

            # Determinando a direção de movimento do mouse
            target_x = current_mouse_x
            target_y = current_mouse_y

            if head_x < w * 0.4:  # Direita
                target_x -= speed
            elif head_x > w * 0.6:  # Esquerda
                target_x += speed

            if head_y < h * 0.4:  # Cima
                target_y -= speed
            elif head_y > h * 0.6:  # Baixo
                target_y += speed

            # Suavização do movimento
            current_mouse_x += (target_x - current_mouse_x) * smooth_factor
            current_mouse_y += (target_y - current_mouse_y) * smooth_factor
            
            # Move o mouse
            pyautogui.moveTo(current_mouse_x, current_mouse_y)

            # Desenha um círculo na posição do nariz
            cv2.circle(frame, (head_x, head_y), 5, (0, 255, 0), -1)

            # Detecção de piscadas
            left_eye_top = face_landmarks.landmark[159]
            left_eye_bottom = face_landmarks.landmark[145]
            right_eye_top = face_landmarks.landmark[386]
            right_eye_bottom = face_landmarks.landmark[374]
            
            left_eye_height = abs(left_eye_top.y - left_eye_bottom.y)
            right_eye_height = abs(right_eye_top.y - right_eye_bottom.y)

            # Inicializa as alturas iniciais dos olhos, se necessário
            if initial_left_eye_height is None:
                initial_left_eye_height = left_eye_height
            if initial_right_eye_height is None:
                initial_right_eye_height = right_eye_height

            min_eye_closure = 0.01  # Ajuste conforme necessário
            
            # Verifica se a altura dos olhos está reduzida significativamente (indicativo de piscar)
            if (initial_left_eye_height - left_eye_height > min_eye_closure or
                initial_right_eye_height - right_eye_height > min_eye_closure):
                
                eyes_blinked = True
                pyautogui.click()  # Simula o clique do mouse

            # Atualiza as alturas iniciais se o piscar ainda não foi detectado
            if not eyes_blinked:
                initial_left_eye_height = left_eye_height
                initial_right_eye_height = right_eye_height
            
            # Reseta o status de piscada
            if eyes_blinked:
                eyes_blinked = False  # Reseta para próxima detecção

    # Verifica se as teclas estão sendo pressionadas para ajustar a velocidade
    if keyboard.is_pressed('up') and speed < max_speed:
        speed += 1  # Aumenta a velocidade
    elif keyboard.is_pressed('down') and speed > min_speed:
        speed -= 1  # Diminui a velocidade
    elif keyboard.is_pressed('right') and speed < max_speed:
        smooth_factor += 0.1  # Aumenta o fator de virada
    elif keyboard.is_pressed('left') and speed > min_speed:
        smooth_factor -= 0.1  # Diminui o fator de virada

    # Exibe a imagem
    cv2.imshow('Head Controlled Mouse', frame)

    # Encerra ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
