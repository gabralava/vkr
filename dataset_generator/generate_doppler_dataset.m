% =========================================================================
% ГЕНЕРАТОР ДАТАСЕТА ДЛЯ МОДЕЛИ 3D U-Net
% =========================================================================

function generate_doppler_dataset()
    % --- Настройки генерации ---
    num_samples = 100;       % Количество новых пар для генерации за запуск
    output_dir = 'dataset_h5_2';
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    % Акустические константы
    c = 1540;              % Скорость звука (м/с)
    fs = 100e6;            % Частота дискретизации Field II (100 МГц)
    f0 = 5e6;              % Фиксированная клиническая частота (5 МГц)
    lambda = c / f0;       % Длина волны
    
    C = 2;                 % Каналы (I и Q)
    T_frames = 32;         % Ансамбль (Slow time)
    X_lines = 64;          % Лучи
    Y_depth = 64;          % Отсчеты по глубине (высота)
    
    pitch = 0.3e-3;        % Шаг элементов датчика (300 мкм)

    % Проверка и создание пула параллельных вычислений
    poolobj = gcp('nocreate');
    if isempty(poolobj)
        parpool(maxNumCompThreads);
    end

    % =====================================================================
    % Создание файлов
    % =====================================================================
    existing_files = dir(fullfile(output_dir, 'sample_*.h5'));
    if isempty(existing_files)
        last_idx = 0;
    else
        indices = zeros(length(existing_files), 1);
        for f = 1:length(existing_files)
            tokens = regexp(existing_files(f).name, 'sample_(\d+)\.h5', 'tokens');
            if ~isempty(tokens)
                indices(f) = str2double(tokens{1}{1});
            end
        end
        last_idx = max(indices);
    end
    
    % Формируем список уникальных имен для каждого параллельного воркера
    file_names = cell(num_samples, 1);
    for i = 1:num_samples
        file_names{i} = fullfile(output_dir, sprintf('sample_%04d.h5', last_idx + i));
    end

    fprintf('Начинаем генерацию. Файлы будут сохранены начиная с индекса %d...\n', last_idx + 1);

    % =====================================================================
    % PARFOR LOOP
    % =====================================================================
    parfor i = 1:num_samples
        % Инициализация Field II на каждом изолированном ядре
        field_init(-1);
        set_field('show_times', 0); 
        set_field('c', c);
        set_field('fs', fs);
        set_field('use_rectangles', 1);

        % --- Клиническая рандомизация ---
        prf = 3000 + rand() * 3000;                 % PRF: 3000 - 6000 Гц
        center_x = (-4 + rand() * 8) * 1e-3;        % Смещение сосуда X: -4..+4 мм
        center_z = (25 + rand() * 10) * 1e-3;       % Глубина сосуда Z: 25..35 мм
        vessel_radius = (2.0 + rand() * 2.5) * 1e-3;% Радиус сосуда: 2.0..4.5 мм
        theta_vessel = pi/5 + rand() * (pi/4);      % Угол наклона сосуда: 36..81 градусов
        
        flow_dir = sign(randn());                   % Направление потока (+1 / -1)
        v_blood_max = flow_dir * (0.1 + rand() * 0.4); % Скорость крови: 0.1..0.5 м/с
        v_clutter_max = 0.005 + rand() * 0.005;     % Пульсация стенок: 5..10 мм/с
        clutter_freq = 1.0 + rand() * 1.0;          % Частота пульсации (ЧСС): 1..2 Гц (60-120 уд/мин)
        
        cbr_db = 40 + rand() * 20;                  % Энергетический контраст Клаттер/Кровь: 40-60 дБ
        snr_db = 10 + rand() * 10;                  % Шум относительно крови: 10-20 дБ
        
        focal_depth = center_z + (randn() * 3e-3);  % Фокус УЗИ-луча
        num_cycles = 1.5 + rand() * 1.0;            % Длина импульса (разрешение)

        % Создание апертур датчика
        emit_aperture = xdc_linear_array(X_lines, pitch*0.9, 5e-3, pitch*0.1, 1, 1, [0 0 focal_depth]);
        receive_aperture = xdc_linear_array(X_lines, pitch*0.9, 5e-3, pitch*0.1, 1, 1, [0 0 focal_depth]);
        
        t_impulse = 0:1/fs:(num_cycles/f0);
        impulse_response = sin(2*pi*f0*t_impulse) .* hanning(length(t_impulse))';
        xdc_impulse(emit_aperture, impulse_response);
        xdc_impulse(receive_aperture, impulse_response);

        % --- Распределение рассеивателей ---
        x_start = -11e-3; x_end = 11e-3;
        z_start = center_z - 8e-3; z_end = center_z + 8e-3;
        
        N_scatterers = 6000; 
        x_pos = x_start + rand(N_scatterers, 1) * (x_end - x_start);
        y_pos = (rand(N_scatterers, 1) - 0.5) * 2e-3; 
        z_pos = z_start + rand(N_scatterers, 1) * (z_end - z_start);
        positions = [x_pos, y_pos, z_pos];

        vessel_dir = [sin(theta_vessel), 0, cos(theta_vessel)];
        center_point = [center_x, 0, center_z];
        vecs = positions - center_point;
        projs = sum(vecs .* vessel_dir, 2);
        dists = sqrt(sum(vecs.^2, 2) - projs.^2);
        
        blood_idx = dists <= vessel_radius;
        tissue_idx = ~blood_idx;

        pos_blood = positions(blood_idx, :);
        pos_tissue = positions(tissue_idx, :);
        dists_blood = dists(blood_idx);
        
        amp_blood = randn(sum(blood_idx), 1);
        clutter_scale = 10^(cbr_db / 20);
        amp_tissue = randn(sum(tissue_idx), 1) * clutter_scale; 

        % Выделение памяти
        rf_blood_full = cell(T_frames, X_lines);
        rf_tissue_t1 = cell(1, X_lines); % Ткани считаем только для T=1!
        
        N_active = 24; % Субапертура датчика
        
        % =================================================================
        % 5. СИМУЛЯЦИЯ (СМЕШАННЫЙ РЕЖИМ)
        % =================================================================
        % Шаг А: Симуляция тканей только на первом кадре (T=1)
        for x_line = 1:X_lines
            x_focus = -9e-3 + (x_line-1) * (18e-3 / (X_lines-1));
            
            apo = zeros(1, X_lines);
            start_el = max(1, round(x_line - N_active/2));
            end_el = min(X_lines, round(x_line + N_active/2));
            apo(start_el:end_el) = 1; 
            
            xdc_apodization(emit_aperture, 0, apo);
            xdc_apodization(receive_aperture, 0, apo);
            xdc_center_focus(emit_aperture, [x_focus 0 0]);
            xdc_focus(emit_aperture, 0, [x_focus 0 focal_depth]);
            xdc_center_focus(receive_aperture, [x_focus 0 0]);
            xdc_focus(receive_aperture, 0, [x_focus 0 focal_depth]);
            
            [v_t, t_start_t] = calc_scat(emit_aperture, receive_aperture, pos_tissue, amp_tissue);
            rf_tissue_t1{x_line} = struct('v', v_t, 't0', t_start_t);
        end
        
        % Шаг Б: Симуляция движущейся крови во всех 32 кадрах
        for t = 1:T_frames
            t_slow = (t - 1) / prf;
            v_blood_current = v_blood_max * (1 - (dists_blood / vessel_radius).^2);
            dx_blood = v_blood_current .* sin(theta_vessel) * t_slow;
            dz_blood = v_blood_current .* cos(theta_vessel) * t_slow;
            cur_pos_blood = pos_blood + [dx_blood, zeros(size(dx_blood)), dz_blood];
            
            for x_line = 1:X_lines
                x_focus = -9e-3 + (x_line-1) * (18e-3 / (X_lines-1));
                
                apo = zeros(1, X_lines);
                start_el = max(1, round(x_line - N_active/2));
                end_el = min(X_lines, round(x_line + N_active/2));
                apo(start_el:end_el) = 1; 
                
                xdc_apodization(emit_aperture, 0, apo);
                xdc_apodization(receive_aperture, 0, apo);
                xdc_center_focus(emit_aperture, [x_focus 0 0]);
                xdc_focus(emit_aperture, 0, [x_focus 0 focal_depth]);
                xdc_center_focus(receive_aperture, [x_focus 0 0]);
                xdc_focus(receive_aperture, 0, [x_focus 0 focal_depth]);
                
                [v_b, t_start_b] = calc_scat(emit_aperture, receive_aperture, cur_pos_blood, amp_blood);
                rf_blood_full{t, x_line} = struct('v', v_b, 't0', t_start_b);
            end
        end
        
        % Освобождение ресурсов Field II
        xdc_free(emit_aperture);
        xdc_free(receive_aperture);
        field_end();
        
        % =================================================================
        % 6. DEMODULATION И АНАЛИТИЧЕСКАЯ ФАЗОВАЯ ИНТЕРПОЛЯЦИЯ КЛАТТЕРА
        % =================================================================
        IQ_blood = zeros(Y_depth, X_lines, T_frames);
        IQ_tissue = zeros(Y_depth, X_lines, T_frames);
        
        patch_height = 12e-3; 
        jitter = (rand()-0.5) * 4e-3; 
        window_center = center_z + jitter;
        
        target_t_start = 2 * (window_center - patch_height/2) / c;
        target_t_end   = 2 * (window_center + patch_height/2) / c;
        t_axis = linspace(target_t_start, target_t_end, Y_depth)';
        
        demod_signal = exp(-1j * 2 * pi * f0 * t_axis);
        
        % Расчет смещения тканей для каждого кадра во времени
        % dz(t) = (v_clutter_max / (2*pi*f_clutter)) * (1 - cos(2*pi*f_clutter*t_slow))
        dz_tissue_frames = zeros(T_frames, 1);
        for t = 1:T_frames
            t_slow = (t - 1) / prf;
            dz_tissue_frames(t) = (v_clutter_max / (2*pi*clutter_freq)) * (1 - cos(2*pi*clutter_freq*t_slow));
        end
        
        for t = 1:T_frames
            % Фазовый сдвиг для когерентного клаттера на текущем кадре
            % delta_phi = - (4*pi / lambda) * dz
            phase_shift = exp(-1j * (4 * pi / lambda) * dz_tissue_frames(t));
            
            for x_line = 1:X_lines
                % Кровь (Честная демодуляция)
                sb = rf_blood_full{t, x_line};
                if ~isempty(sb.v)
                    t_vec = sb.t0 + (0:length(sb.v)-1)'/fs;
                    rf_interp = interp1(t_vec, sb.v, t_axis, 'linear', 0);
                    IQ_blood(:, x_line, t) = rf_interp .* demod_signal;
                end
                
                % Ткань (Аналитическое фазовое смещение от T=1)
                st = rf_tissue_t1{x_line};
                if ~isempty(st.v)
                    t_vec = st.t0 + (0:length(st.v)-1)'/fs;
                    rf_interp = interp1(t_vec, st.v, t_axis, 'linear', 0);
                    % Применяем фазовый сдвиг когерентного движения стенок
                    IQ_tissue(:, x_line, t) = (rf_interp .* demod_signal) * phase_shift;
                end
            end
        end
        
        % Векторизованный ФНЧ
        cutoff_freq = f0 * 0.4;
        [b_filt, a_filt] = butter(2, cutoff_freq / (fs/2), 'low');
        try
            IQ_blood = filtfilt(b_filt, a_filt, IQ_blood);
            IQ_tissue = filtfilt(b_filt, a_filt, IQ_tissue);
        catch
            IQ_blood = zeros(Y_depth, X_lines, T_frames);
            IQ_tissue = zeros(Y_depth, X_lines, T_frames);
        end
        
        % =================================================================
        % 7. ДОБАВЛЕНИЕ ШУМА И ЛОГАРИФМИЧЕСКАЯ НОРМАЛИЗАЦИЯ
        % =================================================================
        Target_Complex = IQ_blood;
        power_blood = var(Target_Complex(:));
        if power_blood == 0, power_blood = 1e-10; end 
        
        power_noise = power_blood / (10^(snr_db / 10));
        noise_complex = sqrt(power_noise/2) * (randn(size(Target_Complex)) + 1j * randn(size(Target_Complex)));
        
        Input_Complex = IQ_blood + IQ_tissue + noise_complex;
        
        % Нормализация по максимальному значению Input
        max_val = max(abs(Input_Complex(:)));
        if max_val == 0, max_val = 1; end
        
        Input_Norm = Input_Complex ./ max_val;
        Target_Norm = Target_Complex ./ max_val;
        
        % Логарифмическая компрессия (Сжатие динамического диапазона 60 дБ с сохранением фазы)
        mu = 1000;
        Input_Norm = (log1p(mu * abs(Input_Norm)) / log1p(mu)) .* exp(1j * angle(Input_Norm));
        Target_Norm = (log1p(mu * abs(Target_Norm)) / log1p(mu)) .* exp(1j * angle(Target_Norm));
        
        % Подготовка 4D тензоров [Y_depth, X_lines, T_frames, C] = [64, 64, 32, 2]
        % (Python прочитает это как [C, T, X, Y] = [2, 32, 64, 64])
        Input_Tensor = zeros(Y_depth, X_lines, T_frames, C, 'single');
        Input_Tensor(:, :, :, 1) = real(Input_Norm); 
        Input_Tensor(:, :, :, 2) = imag(Input_Norm); 
        
        Target_Tensor = zeros(Y_depth, X_lines, T_frames, C, 'single');
        Target_Tensor(:, :, :, 1) = real(Target_Norm);
        Target_Tensor(:, :, :, 2) = imag(Target_Norm);
        
        % Запись HDF5
        filename = file_names{i};
        h5create(filename, '/input', size(Input_Tensor), 'Datatype', 'single');
        h5create(filename, '/target', size(Target_Tensor), 'Datatype', 'single');
        h5write(filename, '/input', Input_Tensor);
        h5write(filename, '/target', Target_Tensor);
        
        fprintf('Семпл %s успешно сгенерирован (CBR: %.1f dB, PRF: %.1f Hz)\n', ...
            filename, cbr_db, prf);
    end
    
    fprintf('Генерация успешно завершена! Все данные в папке "%s"\n', output_dir);
end