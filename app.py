"""
app.py — MODE Pipeline · Emanuel Duarte · 2026
ε dinámico · τ semidinamico · R³ descriptor de co-estabilización
REVISIÓN v2.0: Valores HONESTOS sin inflación · Corre SIN BUGS en Streamlit
"""
import warnings; warnings.filterwarnings('ignore')
import traceback, io
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import streamlit as st
from pipeline import AttractorPipeline, _logistic_map

AUTHOR  = "Investigador: Emanuel Duarte"
VERSION = "v2.0 · Pergamino, Argentina · 2026 (Revisión honesta)"

P = {
    'bg':'#080c10','surface':'#0f1419','card':'#141b23','border':'#1e2a36',
    'text':'#cdd9e5','muted':'#768390','accent':'#4a9eff','green':'#2dbe6c',
    'red':'#e5534b','orange':'#d4a017','purple':'#a371f7','teal':'#2fb392',
}

plt.rcParams.update({
    'figure.facecolor':P['bg'],'axes.facecolor':P['surface'],
    'axes.edgecolor':P['border'],'axes.labelcolor':P['text'],
    'xtick.color':P['muted'],'ytick.color':P['muted'],
    'text.color':P['text'],'grid.color':P['border'],'grid.alpha':0.5,
    'font.family':'monospace','font.size':9,
    'axes.spines.top':False,'axes.spines.right':False,
})

# ── SEÑALES DEMO ────────────────────────────────────────────
def lorenz_ts(n=1000):
    x,y,z=1.0,1.0,1.05; o=[]
    for _ in range(n):
        dx=10*(y-x);dy=x*(28-z)-y;dz=x*y-(8/3)*z
        x+=dx*.01;y+=dy*.01;z+=dz*.01;o.append(x)
    return np.array(o)

def rossler_ts(n=1000):
    x,y,z=1.0,0.0,0.0; o=[]
    for _ in range(n):
        dx=-y-z;dy=x+0.2*y;dz=0.2+z*(x-5.7)
        x+=dx*.05;y+=dy*.05;z+=dz*.05;o.append(x)
    return np.array(o)

DEMOS = {
    'Lorenz (caótico clásico)':           lorenz_ts,
    'Rössler (caos débil)':               rossler_ts,
    'Mapa Logístico r=3.9 (caótico)':    lambda n=1000:_logistic_map(n,r=3.9),
    'Mapa Logístico r=3.5 (periódico)':  lambda n=1000:_logistic_map(n,r=3.5),
    'Mapa Logístico r=3.7 (transición)': lambda n=1000:_logistic_map(n,r=3.7),
    'Senoidal + ruido bajo':             lambda n=1000:np.sin(2*np.pi*0.05*np.arange(n))+0.05*np.random.default_rng(0).normal(size=n),
    'Senoidal + ruido alto':             lambda n=1000:np.sin(2*np.pi*0.05*np.arange(n))+0.5*np.random.default_rng(1).normal(size=n),
    'Ruido blanco':                       lambda n=1000:np.random.default_rng(2).normal(size=n),
    'Ruido rosa (1/f)':                   lambda n=1000:np.cumsum(np.random.default_rng(3).normal(size=n)),
}

# ── UTILIDADES ──────────────────────────────────────────────
def watermark(fig):
    fig.text(0.5,0.5,AUTHOR,fontsize=10,color='white',alpha=0.07,
             ha='center',va='center',rotation=28,transform=fig.transFigure)
    fig.text(0.99,0.01,AUTHOR,fontsize=6.5,color='white',alpha=0.25,
             ha='right',va='bottom',transform=fig.transFigure)

def to_png(fig,dpi=115):
    watermark(fig)
    buf=io.BytesIO()
    fig.savefig(buf,format='png',dpi=dpi,bbox_inches='tight',facecolor=fig.get_facecolor())
    buf.seek(0);plt.close(fig);return buf.read()

def compute_baselines(x,result):
    fv=np.abs(np.fft.rfft(x-x.mean()))
    psd=fv**2;psdn=psd/(psd.sum()+1e-12)
    h_spec=float(-np.sum(psdn[psdn>0]*np.log2(psdn[psdn>0]))/np.log2(max(len(psdn),2)))
    hist,_=np.histogram(x,bins=32,density=True);hn=hist/(hist.sum()+1e-12)
    h_shan=float(-np.sum(hn[hn>0]*np.log2(hn[hn>0]))/np.log2(32))
    d2=result['metrics'].get('D2') or 0.0
    return h_spec,h_shan,d2

# ── FIGURAS ─────────────────────────────────────────────────
def fig_signal(x,label):
    fig,axes=plt.subplots(1,2,figsize=(10,2.8),facecolor=P['bg'])
    ax=axes[0];ax.set_facecolor(P['surface'])
    ax.plot(x[:600],lw=0.65,color=P['accent'],alpha=0.92)
    ax.fill_between(range(min(600,len(x))),x[:600],alpha=0.07,color=P['accent'])
    ax.set_xlabel('t',fontsize=8);ax.set_ylabel('x(t)',fontsize=8)
    ax.set_title(label[:45],fontsize=8,color=P['accent']);ax.grid(True,alpha=0.25)
    ax2=axes[1];ax2.set_facecolor(P['surface'])
    fv=np.abs(np.fft.rfft(x-x.mean()));fr=np.fft.rfftfreq(len(x))
    ax2.semilogy(fr[1:],fv[1:],color=P['purple'],lw=0.65,alpha=0.88)
    ax2.fill_between(fr[1:],fv[1:],alpha=0.06,color=P['purple'])
    ax2.set_xlabel('Frecuencia',fontsize=8);ax2.set_ylabel('|FFT|',fontsize=8)
    ax2.set_title('Espectro de potencia',fontsize=8,color=P['accent']);ax2.grid(True,alpha=0.25)
    fig.tight_layout();return fig

def fig_epsilon(result):
    eps=result['epsilon_series']
    fig,ax=plt.subplots(figsize=(10,2.4),facecolor=P['bg'])
    ax.set_facecolor(P['surface'])
    t=np.arange(len(eps))
    ax.fill_between(t,eps,alpha=0.18,color=P['teal'])
    ax.plot(t,eps,color=P['teal'],lw=0.75,alpha=0.9)
    med=np.median(eps)
    ax.axhline(med,color=P['orange'],lw=1.2,ls='--',label=f'ε̃={med:.5f}')
    ax.set_xlabel('t (índice embedding)',fontsize=8);ax.set_ylabel('ε(t)',fontsize=8)
    ax.set_title('ε dinámico · escala local del sistema',fontsize=9,color=P['accent'])
    ax.legend(fontsize=7,framealpha=0.15);ax.grid(True,alpha=0.25)
    fig.tight_layout();return fig

def fig_attractor(result):
    Y=result['embedding'];eps=result['epsilon_series']
    if Y.shape[1]<3:return None
    norm=Normalize(vmin=eps.min(),vmax=eps.max());cmap=plt.cm.plasma
    fig=plt.figure(figsize=(6,5.2),facecolor=P['bg'])
    ax=fig.add_subplot(111,projection='3d',facecolor=P['surface'])
    n=len(Y);step=max(1,n//2000)
    Ys=Y[::step];es=eps[:n][::step]
    ax.scatter(Ys[:,0],Ys[:,1],Ys[:,2],c=es,cmap=cmap,norm=norm,s=1.0,alpha=0.75)
    ax.set_xlabel('y(t)',fontsize=7,color=P['text'])
    ax.set_ylabel('y(t-τ)',fontsize=7,color=P['text'])
    ax.set_zlabel('y(t-2τ)',fontsize=7,color=P['text'])
    ax.set_title(f"τ={result['tau']} · ε={result['epsilon']:.5f}",color=P['accent'],fontsize=9,pad=8)
    for pane in [ax.xaxis.pane,ax.yaxis.pane,ax.zaxis.pane]:
        pane.fill=False;pane.set_edgecolor(P['border'])
    ax.tick_params(colors=P['border'],labelsize=5)
    sm=ScalarMappable(cmap=cmap,norm=norm);sm.set_array([])
    cb=fig.colorbar(sm,ax=ax,shrink=0.45,pad=0.1)
    cb.ax.tick_params(labelsize=5,colors=P['text']);cb.set_label('ε(t)',color=P['text'],fontsize=6)
    fig.tight_layout();return fig

def fig_metrics(result):
    r3=result['R3'];sm=r3['stability_map'];delta=r3['delta']
    keys=['lambda','D2','LZ','TE']
    labels=['λ (Lyapunov)','D₂ (Dim. Corr.)','C_LZ (Compl.)','TE (Trans. Ent.)']
    grads=[sm.get(k,{}).get('gradient',0.0) for k in keys]
    stable=[sm.get(k,{}).get('stable',False) for k in keys]
    colors=[P['green'] if s else P['red'] for s in stable]
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),facecolor=P['bg'])
    ax=axes[0];ax.set_facecolor(P['surface'])
    bars=ax.barh(labels,grads,color=colors,alpha=0.85,height=0.52)
    ax.axvline(delta,color=P['orange'],lw=1.8,ls='--',label=f'δ={delta} ({r3["regime"]})')
    ax.set_xlabel('|∂μ/∂τ| normalizado (RMS)',fontsize=7)
    ax.set_title('Sensibilidad a τ por métrica',fontsize=9,color=P['accent'])
    ax.legend(fontsize=7,framealpha=0.15);ax.grid(True,axis='x',alpha=0.25)
    mx=max(grads) if max(grads)>0 else 0.01
    for bar,g in zip(bars,grads):
        ax.text(g+mx*0.03,bar.get_y()+bar.get_height()/2,f'{g:.6f}',
                va='center',ha='left',fontsize=6,color=P['text'])
    ax2=axes[1];ax2.set_facecolor(P['surface'])
    theta=np.linspace(np.pi,0,300)
    for i in range(len(theta)-1):
        c=plt.cm.RdYlGn(i/(len(theta)-1))
        ax2.fill_between([np.cos(theta[i]),np.cos(theta[i+1])],
                         [np.sin(theta[i])*0.68,np.sin(theta[i+1])*0.68],
                         [np.sin(theta[i])*1.0,np.sin(theta[i+1])*1.0],color=c,alpha=0.88)
    score=r3['R3_score'];angle=np.pi*(1-score)
    ax2.annotate('',xy=(0.83*np.cos(angle),0.83*np.sin(angle)),xytext=(0,0),
                 arrowprops=dict(arrowstyle='->',color=P['text'],lw=2.8),zorder=5)
    ax2.plot(0,0,'o',color=P['text'],ms=6,zorder=6)
    coh_c=P['green'] if r3['coherent'] else P['red']
    coh_t='✔  COHERENTE' if r3['coherent'] else '���  NO COHERENTE'
    ax2.text(0,-0.20,f"R³ = {score:.7f}",ha='center',fontsize=14,fontweight='bold',color=P['accent'])
    ax2.text(0,-0.44,coh_t,ha='center',fontsize=10,color=coh_c)
    ax2.text(0,-0.64,r3['regime_desc'],ha='center',fontsize=8,color=P['orange'])
    ax2.text(-1.04,-0.1,'0',ha='center',fontsize=8,color=P['red'])
    ax2.text(1.04,-0.1,'1',ha='center',fontsize=8,color=P['green'])
    ax2.text(0,1.07,'0.5',ha='center',fontsize=8,color=P['orange'])
    ax2.set_xlim(-1.28,1.28);ax2.set_ylim(-0.88,1.22)
    ax2.set_aspect('equal');ax2.axis('off')
    ax2.set_title('R³ Score · co-estabilización (CONTINUO)',fontsize=9,color=P['accent'])
    fig.tight_layout(pad=1.5);return fig

def fig_baselines(x,result):
    h_spec,h_shan,d2=compute_baselines(x,result)
    r3=result['R3']['R3_score'];coherent=result['R3']['coherent']
    fig,ax=plt.subplots(figsize=(8,3.2),facecolor=P['bg'])
    ax.set_facecolor(P['surface'])
    methods=['H Fourier','H Shannon','D₂ (norm)','R³']
    vals=[h_spec,h_shan,min(d2/3.0,1.0),r3]
    bar_c=[P['muted'],P['muted'],P['muted'],P['green'] if coherent else P['red']]
    bars=ax.bar(methods,vals,color=bar_c,alpha=0.85,width=0.55)
    ax.axhline(0.60,color=P['orange'],lw=1.2,ls='--',alpha=0.7,label='Umbral R³=0.60')
    ax.set_ylim(0,1.15);ax.set_ylabel('Valor normalizado',fontsize=8)
    ax.set_title('Comparación: baselines vs R³',fontsize=9,color=P['accent'])
    ax.legend(fontsize=7,framealpha=0.15);ax.grid(True,axis='y',alpha=0.25)
    for bar,v in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2,v+0.02,f'{v:.6f}',
                ha='center',fontsize=8,color=P['text'])
    fig.tight_layout();return fig

def fig_windowed(sig,win_results):
    if not win_results:return None
    t_v=[r['t_s'] for r in win_results]
    r3_v=[r['R3'] for r in win_results]
    coh_v=[r['coherente'] for r in win_results]
    tau_v=[r['tau'] for r in win_results]
    fig,axes=plt.subplots(3,1,figsize=(11,7),facecolor=P['bg'])
    fig.suptitle('Análisis por ventanas temporales',color=P['accent'],fontsize=11,fontweight='bold')
    ax1=axes[0];ax1.set_facecolor(P['surface'])
    ax1.plot(sig[:min(3000,len(sig))],lw=0.5,color=P['accent'],alpha=0.9)
    ax1.set_ylabel('Amplitud',fontsize=8);ax1.set_title('Señal',fontsize=8,color=P['text']);ax1.grid(True,alpha=0.2)
    ax2=axes[1];ax2.set_facecolor(P['surface'])
    bc=[P['green'] if c else P['red'] for c in coh_v]
    w=max((t_v[1]-t_v[0])*0.8 if len(t_v)>1 else 1,0.5)
    ax2.bar(t_v,r3_v,width=w,color=bc,alpha=0.85)
    ax2.axhline(0.60,color=P['orange'],lw=1.2,ls='--',alpha=0.8)
    ax2.set_ylim(0,1.15);ax2.set_ylabel('R³',fontsize=8)
    ax2.set_title('R³ por ventana (verde=coherente  rojo=incoherente)',fontsize=8,color=P['text']);ax2.grid(True,alpha=0.2)
    ax3=axes[2];ax3.set_facecolor(P['surface'])
    ax3.plot(t_v,tau_v,color=P['purple'],lw=1.2,marker='o',ms=3,alpha=0.9)
    ax3.fill_between(t_v,tau_v,alpha=0.1,color=P['purple'])
    ax3.set_xlabel('t (unidades de índice)',fontsize=8);ax3.set_ylabel('τ',fontsize=8)
    ax3.set_title('Evolución de τ semidinamico',fontsize=8,color=P['text']);ax3.grid(True,alpha=0.2)
    fig.tight_layout(rect=[0,0,1,0.95]);return fig

# ── LECTOR MIT-BIH ──────────────────────────────────────────
def read_mitbih_bytes(dat_file,hea_file):
    lines=hea_file.read().decode('latin-1').strip().split('\n')
    hdr=lines[0].split();fs,n=int(hdr[2]),int(hdr[3])
    si=lines[1].split();gain,bl=float(si[2]),int(si[4])
    raw=dat_file.read();s=[];i=0
    while i+2<len(raw):
        b0,b1,b2=raw[i],raw[i+1],raw[i+2]
        s1=b0|((b1&0x0F)<<8);s1=s1-4096 if s1>=2048 else s1
        s2=b2|((b1&0xF0)<<4);s2=s2-4096 if s2>=2048 else s2
        s.extend([s1,s2]);i+=3
    return (np.array(s[::2][:n])-bl)/gain,fs,n

# ════════════════════════════════════���═════════════════════════
# APP PRINCIPAL
# ══════════════════════════════════════════════════════════════
def main():
    st.set_page_config(page_title="MODE · Attractor Pipeline",page_icon="🌀",
                       layout="wide",initial_sidebar_state="expanded")
    st.markdown(f"""<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');
    html,body,[class*="css"]{{font-family:'Space Grotesk',sans-serif;background:{P['bg']};color:{P['text']};}}
    [data-testid="stAppViewContainer"]{{background:{P['bg']};}}
    [data-testid="stSidebar"]{{background:{P['surface']};border-right:1px solid {P['border']};}}
    [data-testid="stSidebar"] *{{color:{P['text']} !important;}}
    .stButton>button{{background:linear-gradient(135deg,#1a4a7a,#0d2d52);color:#4a9eff;
        border:1px solid #1e3a5a;border-radius:6px;font-family:'JetBrains Mono',monospace;
        font-weight:600;letter-spacing:.05em;transition:all .2s ease;}}
    .stButton>button:hover{{background:linear-gradient(135deg,#1e5a9a,#1a3d6a);
        border-color:#4a9eff;color:#80bcff;}}
    .stMetric{{background:{P['card']};border:1px solid {P['border']};border-radius:8px;padding:12px;}}
    .stMetric label{{color:{P['muted']} !important;font-size:.78rem !important;}}
    h1,h2,h3{{font-family:'Space Grotesk',sans-serif;color:{P['accent']};}}
    .stTabs [data-baseweb="tab"]{{font-family:'JetBrains Mono',monospace;color:{P['muted']};border-bottom:2px solid transparent;}}
    .stTabs [aria-selected="true"]{{color:{P['accent']} !important;border-bottom:2px solid {P['accent']} !important;}}
    </style>""",unsafe_allow_html=True)

    # HEADER
    col_t,col_i=st.columns([3,1])
    with col_t:
        st.markdown("# 🌀 MODE Pipeline")
        st.markdown("**H1** ε dinámico &nbsp;·&nbsp; **H2** τ semidinamico &nbsp;·&nbsp; **H3** R³ co-estabilización")
    with col_i:
        st.markdown(f"<div style='text-align:right;padding-top:12px;color:{P['muted']};font-family:JetBrains Mono,monospace;font-size:.74rem;'>{AUTHOR}<br>{VERSION}</div>",unsafe_allow_html=True)
    st.divider()

    # SIDEBAR
    with st.sidebar:
        st.markdown("### ⚙️ Configuración")
        st.divider()
        modo=st.radio("Modo",["🔬 Señal sintética","📁 Cargar CSV","❤️ ECG MIT-BIH","📊 Ventanas temporales"])
        st.divider()
        st.markdown("**Pipeline**")
        m_dim=st.slider("Dimensión embedding m",2,6,3)
        max_tau=st.slider("τ máximo (AMI)",10,80,40,5)

        x_data,label,extra={},{},'sin datos'

        if modo=="🔬 Señal sintética":
            st.divider(); st.markdown("**Señal**")
            demo_name=st.selectbox("Seleccionar",list(DEMOS.keys()))
            N=st.slider("N muestras",300,3000,1000,100)
            x_data=DEMOS[demo_name](N); label=demo_name

        elif modo=="📁 Cargar CSV":
            st.divider(); st.markdown("**Archivo**")
            up_csv=st.file_uploader("CSV (una columna)",type=['csv','txt'])
            if up_csv:
                x_data=pd.read_csv(up_csv,header=None).iloc[:,0].dropna().values.astype(float)
                label=up_csv.name

        elif modo=="❤️ ECG MIT-BIH":
            st.divider(); st.markdown("**Archivos ECG**")
            st.caption("Subí el .dat y el .hea del mismo registro")
            up_dat=st.file_uploader("Archivo .dat",type=['dat'])
            up_hea=st.file_uploader("Archivo .hea",type=['hea'])
            ecg_start=st.slider("Inicio (s)",0,600,0,5)
            ecg_dur=st.slider("Duración (s)",5,60,10,5)
            if up_dat and up_hea:
                sig_full,fs_ecg,_=read_mitbih_bytes(up_dat,up_hea)
                x_data=sig_full[int(ecg_start*fs_ecg):int((ecg_start+ecg_dur)*fs_ecg)]
                label=f"ECG t={ecg_start}-{ecg_start+ecg_dur}s"
                extra={'fs':fs_ecg,'sig_full':sig_full}

        elif modo=="📊 Ventanas temporales":
            st.divider(); st.markdown("**Fuente**")
            w_src=st.radio("",["Demo","CSV"],key='wsrc')
            if w_src=="Demo":
                w_demo=st.selectbox("Señal",list(DEMOS.keys()),key='wdm')
                w_N=st.slider("N total",1000,5000,2000,100,key='wN')
                x_data=DEMOS[w_demo](w_N); label=w_demo
            else:
                w_csv=st.file_uploader("CSV",type=['csv','txt'],key='wcsv')
                if w_csv:
                    x_data=pd.read_csv(w_csv,header=None).iloc[:,0].dropna().values.astype(float)
                    label=w_csv.name
            st.divider(); st.markdown("**Ventanas**")
            win_size=st.slider("Tamaño (muestras)",200,2000,1000,100)
            win_step=st.slider("Paso (muestras)",100,1000,500,50)
            win_max=st.slider("Máx. ventanas",5,50,20,5)
            extra={'win_size':win_size,'win_step':win_step,'win_max':win_max}

        st.divider()
        run_btn=st.button("▶ Ejecutar pipeline",type="primary",use_container_width=True)
        st.divider()
        st.markdown(f"""<div style='font-family:JetBrains Mono,monospace;font-size:.72rem;color:{P["muted"]};'>
        <b style='color:{P["text"]}'>δ por régimen (empirical IQR)</b><br><br>
        Estable &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0.06<br>
        Caos débil &nbsp;&nbsp; 0.05<br>
        Caótico &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0.08<br>
        Hipercaótico &nbsp;0.15<br>
        Ruidoso &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0.20<br><br>
        <b style='color:{P["text"]}'>Umbral coherencia</b><br>
        R³ ≥ 0.60 (continuo)</div>""",unsafe_allow_html=True)

    # EJECUCIÓN
    if run_btn:
        if len(x_data)==0:
            st.warning("Cargá datos primero."); st.stop()

        if modo=="📊 Ventanas temporales":
            ws=extra['win_size']; wst=extra['win_step']; wmax=extra['win_max']
            with st.spinner("Analizando ventanas…"):
                pipe=AttractorPipeline(m=m_dim,max_tau=max_tau,verbose=False)
                win_rs=[]; prog=st.progress(0); nw=0
                for start in range(0,len(x_data)-ws,wst):
                    if nw>=wmax: break
                    w=x_data[start:start+ws]
                    if w.std()<1e-6: continue
                    try:
                        r=pipe.run(w,label=f"w{nw}"); r3=r['R3']
                        win_rs.append({'t_s':start,'tau':r['tau'],'epsilon':round(r['epsilon'],5),
                                       'R3':r3['R3_score'],'coherente':r3['coherent'],
                                       'regimen':r3['regime'],'delta':r3['delta']})
                    except Exception as e:
                        st.warning(f"Ventana {nw} falló: {str(e)[:40]}")
                        pass
                    nw+=1; prog.progress(min(nw/wmax,1.0))
                prog.empty()
            st.session_state.update({'mode':modo,'x':x_data,'label':label,
                                     'win_results':win_rs,'result':None})
        else:
            with st.spinner("Calculando ε · τ · métricas · R³…"):
                try:
                    pipe=AttractorPipeline(m=m_dim,max_tau=max_tau,verbose=False)
                    r=pipe.run(x_data,label=label)
                    st.session_state.update({'mode':modo,'x':x_data,'label':label,
                                             'result':r,'win_results':None})
                except Exception as e:
                    st.error(f"Error en pipeline: {e}"); 
                    st.code(traceback.format_exc()); st.stop()

    # PANTALLA INICIAL
    if 'mode' not in st.session_state:
        st.markdown(f"""<div style='text-align:center;padding:60px 20px;'>
        <div style='font-size:3.5rem;margin-bottom:16px;'>🌀</div>
        <h2 style='color:{P["accent"]};font-family:JetBrains Mono,monospace;'>MODE Pipeline v2.0</h2>
        <p style='color:{P["muted"]};font-size:.95rem;max-width:680px;margin:0 auto;line-height:1.7;'>
        Framework de legibilidad observacional para sistemas dinámicos no lineales.<br>
        <b>Revisión 2026-05:</b> Cálculos HONESTOS sin inflación · Gradientes RMS normalizados · R³ continuo<br>
        Seleccioná un modo en el sidebar y presioná <b style='color:{P["text"]}'>▶ Ejecutar pipeline</b>.
        </p><br>
        <p style='color:{P["border"]};font-family:JetBrains Mono,monospace;font-size:.75rem;'>
        {AUTHOR} &nbsp;·&nbsp; {VERSION}</p></div>""",unsafe_allow_html=True)
        st.stop()

    modo_act=st.session_state.get('mode','')

    # ── MODO VENTANAS ────────────────────────────────────────
    if modo_act=="📊 Ventanas temporales":
        win_rs=st.session_state.get('win_results',[])
        sig_w=st.session_state.get('x',np.array([]))
        if not win_rs:
            st.info("Presioná ▶ para analizar."); st.stop()
        coh_n=sum(1 for r in win_rs if r['coherente'])
        r3_med=np.mean([r['R3'] for r in win_rs])
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Ventanas",len(win_rs)); c2.metric("R³ promedio",f"{r3_med:.7f}")
        c3.metric("Coherentes",f"{coh_n} ({coh_n/len(win_rs)*100:.1f}%)")
        c4.metric("Señal",st.session_state.get('label','')[:25])
        st.divider()
        fw=fig_windowed(sig_w,win_rs)
        if fw: st.image(to_png(fw),use_container_width=True)
        st.divider()
        st.subheader("📊 Resultados por ventana")
        df_w=pd.DataFrame(win_rs)
        st.dataframe(df_w,use_container_width=True,hide_index=True)
        st.download_button("⬇ CSV ventanas",df_w.to_csv(index=False).encode(),"ventanas.csv",mime="text/csv")
        st.stop()

    # ── MODOS NORMALES ───────────────────────────────────────
    result=st.session_state.get('result')
    x=st.session_state.get('x',np.array([]))
    label=st.session_state.get('label','')
    if result is None:
        st.info("👈 Presioná ▶ Ejecutar pipeline"); st.stop()

    r3=result['R3']; mvals=result['metrics']

    # Métricas rápidas
    c1,c2,c3,c4,c5,c6=st.columns(6)
    c1.metric("τ semidinamico",result['tau'])
    c2.metric("ε mediana",f"{result['epsilon']:.5f}")
    c3.metric("R³ Score",f"{r3['R3_score']:.7f}",
              delta="✔ coherente" if r3['coherent'] else "✘ no coherente")
    c4.metric("Régimen",r3['regime'])
    c5.metric("δ activo",r3['delta'])
    c6.metric("N",len(x))
    st.divider()

    tab1,tab2,tab3,tab4,tab5=st.tabs([
        "📡 Señal & ε","🌀 Atractor","📐 Métricas & R³","⚖️ Baselines","📋 Export"])

    with tab1:
        st.subheader("Serie temporal y espectro")
        st.image(to_png(fig_signal(x,label)),use_container_width=True)
        st.subheader("ε(t) dinámico")
        st.image(to_png(fig_epsilon(result)),use_container_width=True)

    with tab2:
        fa=fig_attractor(result)
        if fa:
            col_a,col_b=st.columns([2,1])
            with col_a:
                st.subheader("Atractor reconstruido")
                st.image(to_png(fa),use_container_width=True)
            with col_b:
                st.markdown(f"""<div style='background:{P["card"]};border:1px solid {P["border"]};
                border-radius:10px;padding:20px;margin-top:40px;
                font-family:JetBrains Mono,monospace;font-size:.82rem;'>
                <div style='color:{P["muted"]};margin-bottom:8px;'>EMBEDDING</div>
                τ = <b style='color:{P["accent"]};'>{result['tau']}</b><br>
                m = <b style='color:{P["accent"]};'>{result['embedding'].shape[1]}</b><br>
                ε̃ = <b style='color:{P["teal"]};'>{result['epsilon']:.5f}</b><br><br>
                <div style='color:{P["muted"]};margin-bottom:8px;'>RÉGIMEN</div>
                <div style='color:{P["orange"]};'>{r3['regime_desc']}</div><br>
                <div style='color:{P["muted"]};margin-bottom:4px;'>COLOR MAP</div>
                <div style='font-size:.72rem;color:{P["text"]};'>
                azul → zona densa (ε pequeño)<br>amarillo → zona dispersa (ε grande)
                </div></div>""",unsafe_allow_html=True)
        else:
            st.info("Se necesita m≥3 para el atractor 3D.")

    with tab3:
        st.subheader("Sensibilidad a τ y R³ gauge")
        st.image(to_png(fig_metrics(result)),use_container_width=True)
        st.divider()
        st.subheader("Detalle por métrica")
        sm_map=r3['stability_map']
        cols_m=st.columns(4)
        nombres={'lambda':'λ Lyapunov','D2':'D₂ Corr.','LZ':'C_LZ Compl.','TE':'TE Transf.'}
        for i,(k,vd) in enumerate(sm_map.items()):
            with cols_m[i]:
                col=P['green'] if vd['stable'] else P['red']
                st.markdown(f"""<div style='background:{P["card"]};border:1px solid {col}40;
                border-radius:10px;padding:16px;text-align:center;'>
                <div style='color:{P["muted"]};font-size:.75rem;margin-bottom:6px;'>{nombres.get(k,k)}</div>
                <div style='font-size:1.4rem;'>{'✔' if vd['stable'] else '✘'}</div>
                <div style='color:{P["text"]};font-family:JetBrains Mono,monospace;font-size:.75rem;margin-top:6px;'>
                grad={vd['gradient']:.7f}<br>
                <span style='color:{P["muted"]};'>δ={vd['delta']}</span>
                </div></div>""",unsafe_allow_html=True)

    with tab4:
        st.subheader("Comparación con métodos estándar")
        st.image(to_png(fig_baselines(x,result)),use_container_width=True)
        st.divider()
        h_spec,h_shan,d2=compute_baselines(x,result)
        bl_rows=[
            ('H Fourier espectral',f"{h_spec:.7f}",'1=plano','No distingue caos-ruido'),
            ('H Shannon (señal)',  f"{h_shan:.7f}",'1=máx desor','Sin info temporal'),
            ('D₂ clásico',        f"{d2:.7f}",    'Lorenz≈2.05','Escalar puro'),
            ('R³ Score',          f"{r3['R3_score']:.8f}",'≥0.60→coherente','Continuo+régimen'),
            ('Coherente',         '✔' if r3['coherent'] else '✘','—','—'),
            ('Régimen',           r3['regime_desc'],'—','δ semidinamico'),
        ]
        st.dataframe(pd.DataFrame(bl_rows,columns=['Método','Valor','Ref','Nota']),
                     use_container_width=True,hide_index=True)

    with tab5:
        st.subheader("Tabla completa de resultados (Valores HONESTOS sin truncado)")
        rows=[
            ('τ semidinamico',result['tau'],'—'),
            ('ε mediana',f"{result['epsilon']:.8f}",'—'),
            ('Régimen',r3['regime_desc'],'—'),
            ('δ activo',r3['delta'],'—'),
            ('R³ Score',f"{r3['R3_score']:.8f}",'≥0.60 continuo'),
            ('Coherente','✔' if r3['coherent'] else '✘','—'),
            ('λ (Lyapunov)',f"{mvals.get('lambda'):.8f}" if mvals.get('lambda') else 'N/A','<0 estable · >0 caos'),
            ('D₂ (dim. corr.)',f"{mvals.get('D2'):.8f}" if mvals.get('D2') else 'N/A','Lorenz≈2.05'),
            ('C_LZ (compl.)',f"{mvals.get('LZ'):.8f}" if mvals.get('LZ') else 'N/A','0=orden · 1=compl'),
            ('TE (trans. ent.)',f"{mvals.get('TE'):.8f}" if mvals.get('TE') else 'N/A','Flujo inf'),
        ]
        df_res=pd.DataFrame(rows,columns=['Variable','Valor','Referencia'])
        st.dataframe(df_res,use_container_width=True,hide_index=True)
        st.divider()
        ce1,ce2,ce3=st.columns(3)
        with ce1:
            st.download_button("⬇ Resultados CSV",df_res.to_csv(index=False).encode(),"resultados.csv",mime="text/csv")
        with ce2:
            emb=result['embedding']
            cols_=[f'y_t-{i*result["tau"]}' for i in range(emb.shape[1])]
            df_emb=pd.DataFrame(emb,columns=cols_)
            df_emb['epsilon']=result['epsilon_series'][:len(df_emb)]
            st.download_button("⬇ Embedding CSV",df_emb.to_csv(index=False).encode(),"embedding.csv",mime="text/csv")
        with ce3:
            h_spec2,h_shan2,d2_2=compute_baselines(x,result)
            df_bl=pd.DataFrame([('H_Fourier',f"{h_spec2:.8f}"),('H_Shannon',f"{h_shan2:.8f}"),
                                ('D2',f"{d2_2:.8f}"),('R3',f"{r3['R3_score']:.8f}"),
                                ('coherente',r3['coherent']),('regimen',r3['regime'])],
                               columns=['metrica','valor'])
            st.download_button("⬇ Baselines CSV",df_bl.to_csv(index=False).encode(),"baselines.csv",mime="text/csv")

    st.markdown(f"""<div style='text-align:center;padding:10px;margin-top:6px;
    color:{P["border"]};font-family:JetBrains Mono,monospace;font-size:.68rem;'>
    {AUTHOR} &nbsp;·&nbsp; {VERSION}<br>
    Revisión honesta: Gradientes RMS · R³ continuo · Precisión 6-8 decimales
    </div>""",unsafe_allow_html=True)

if __name__=='__main__':
    main()
