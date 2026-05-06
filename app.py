def read_mitbih_safe(record_id):
    # ... (mantené la parte de rutas y carga de bytes igual) ...
    
    # 2. Leer .dat y aplicar ESCUDO
    dat_path = path_base + '.dat'
    with open(dat_path, 'rb') as f:
        raw_bytes = np.frombuffer(f.read(), dtype=np.uint8)
    
    n_groups = len(raw_bytes) // 3 
    b = raw_bytes[:n_groups*3].reshape(-1, 3)
    
    # DESEMPAQUETADO REVISADO (Formato 212 Estándar)
    # Canal 1 (MLII)
    c1 = b[:, 0].astype(np.int16) | ((b[:, 1].astype(np.int16) & 0x0F) << 8)
    c1[c1 >= 2048] -= 4096
    
    # Canal 2 (V5)
    c2 = b[:, 2].astype(np.int16) | ((b[:, 1].astype(np.int16) & 0xF0) << 4)
    c2[c2 >= 2048] -= 4096
    
    # IMPORTANTE: El Pipeline MODE analiza un solo canal por vez para buscar la estructura fractal
    # Usamos c1 que es el canal principal de ECG (MLII)
    return (c1 - baseline) / gain, fs
