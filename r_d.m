function r_d(speed, angle, altitude, number)
    % clear all
    % close all

    % target model
    input_filename = sprintf('回波矩阵_速度=%d_航向角=%d_高度=%d_反射次数=%d.mat', speed, angle, altitude, number);
    load(input_filename);
    % load 回波矩阵.mat
    Rx = Rx.';
    [ns,nb] = size(Rx);
    r = [1:ns];  % range cells
    xr = [1:nb]; % cross-range cells
    bw = Df*ns;  % total bandwidth
    prf = 1/t0;  % prf
    c0 = 2.99792458e8;  % progagation velocity
    Dr = c0/(2*bw); % range resolution
    Rng = Dr*ns;  % total range
    Freq = nb*prf/ns;  % burst repetition freq
    F = prf; % pulse repetition freq
    T = nb*ns*t0; % total time integration

    G0 = fftshift(fft(Rx));
    RG0 = flipud(rot90(G0));

    % display range profiles before range centroid
    %figure
    %colormap(jet(256))
    %imagesc(20*log10(abs(RG0)+eps));
    %ylabel('Pulses')
    %xlabel('Range cells')
    %title('初始距离像')
    %axis('xy')
    %clim = get(gca,'CLim');
    %set(gca,'CLim',clim(2) + [-30 0]);

    %
    % range centroid
    [Xr,Rr,Vr,Ar] = range_centroid(Rx,f0,Df,prf);

    G = fftshift(fft(Xr));
    RG = flipud(rot90(G));

    % display range profiles after range centroid
    %figure
    %colormap(jet(256))
    %imagesc(20*log10(abs(RG)+eps));
    %ylabel('Pulses')
    %xlabel('Range cells')
    %title('距离多普勒后的距离像')
    %axis('xy')
    %clim = get(gca,'CLim');
    %set(gca,'CLim',clim(2) + [-30 0]);
    % display Doppler spectrum from a range profile
    %figure
    %colormap(jet)
    sig = Xr(16,:);
    TF = stft(sig,8);
    %imagesc([0,T],[-F/2,F/2],20*log10(fftshift(abs(TF),1)+eps));
    %xlabel('Time (s)')
    %ylabel('Doppler (Hz)')
    %title('Doppler spectrum after range centroid')
    %axis xy
    %clim = get(gca,'CLim');
    %set(gca,'CLim',clim(2) + [-30 0]);

    % display image after range-centroid
    %figure
    IRG = fftshift(fft(RG),1);
    %colormap(jet(256))
    %imagesc([-Rng/2 Rng/2],[-Freq/2 Freq/2],...
    %    20*log10(abs(IRG)));
    %ylabel('Doppler (Hz)')
    %xlabel('Range (m)')
    %title('Image after Range Centroid')
    %axis('xy')
    %clim = get(gca,'CLim');
    %set(gca,'CLim',clim(2) + [-15 0]);
    %drawnow

    %
    % Doppler centroid
    [Xrd,Xf,Rd,Vd,Ad] = doppler_centroid(Rx,f0,Df,prf,Xr,Rr,Vr,Ar);

    Gd = fftshift(fft(Xrd),1);
    RGd = flipud(rot90(Gd));
    %figure
    %colormap(jet(256))
    %imagesc(20*log10(abs(RGd)+eps));
    %ylabel('Pulses')
    %xlabel('Range cells')
    %title('Range profiles after Doppler Centroid')
    %axis('xy')
    %clim = get(gca,'CLim');
    %set(gca,'CLim',clim(2) + [-30 0]);
    %drawnow

    sigd = Xrd(16,:);
    TFd = stft(sigd,8);
    %figure
    %colormap(jet)
    %imagesc([0,T],[-F/2,F/2],20*log10(abs(TFd)+eps));
    %xlabel('Time (s)')
    %ylabel('Doppler (Hz)')
    %title('Doppler spectrum after Doppler centroid')
    %axis xy
    %clim = get(gca,'CLim');
    %set(gca,'CLim',clim(2) + [-30 0]);
    %drawnow

    %figure
    IRG = fft(RGd);
    %colormap(jet)
    %imagesc([-Rng/2 Rng/2],[-Freq/2 Freq/2],...
    %    20*log10(abs(IRG)));
    %ylabel('Doppler (Hz)')
    %xlabel('Range (m)')
    %title('Image after Doppler Centroid')
    %axis('xy')
    %clim = get(gca,'CLim');
    %set(gca,'CLim',clim(2) + [-30 0]);
    %drawnow

    Gf = fftshift(fft(Xf),1);
    RGf = flipud(rot90(Gf));
    %figure
    %colormap(jet(256))
    %imagesc(20*log10(abs(RGf)+eps));
    %ylabel('Pulses')
    %xlabel('Range cells')
    %title('Range profiles after refining')
    %axis('xy')
    %clim = get(gca,'CLim');
    %set(gca,'CLim',clim(2) + [-30 0]);
    %drawnow

    sig = Xf(16,:);
    TFf = stft(sig,8);
    %figure
    %colormap(jet(256))
    %imagesc([0,T],[-F/2,F/2],20*log10(fftshift(abs(TFf),1)+eps));
    %xlabel('Time (s)')
    %ylabel('Doppler (Hz)')
    %title('Doppler spectrum after Doppler refining')
    %axis xy
    %clim = get(gca,'CLim');
    %set(gca,'CLim',clim(2) + [-30 0]);
    %drawnow

    %figure
    IRGf = fftshift(fft(RGf),1);
    %colormap(jet)
    %imagesc([-Rng/2 Rng/2],[-Freq/2 Freq/2],...
    %    20*log10(abs(IRGf)));
    %ylabel('Doppler (Hz)')
    %xlabel('Range (m)')
    %title('Image after Doppler refining')
    %axis('xy')
    %clim = get(gca,'CLim');
    %set(gca,'CLim',clim(2) + [-30 0]);


    filename = sprintf('Xf_速度=%d_航向角=%d_高度=%d_反射次数=%d.mat', speed, angle, altitude, number);
    save(filename, 'Xf');
    % save('Xf.mat', 'Xf');
end
