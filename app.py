"""
app.py — MODE Pipeline · Emanuel Duarte · 2026
ε dinámico · τ semidinamico · R³ descriptor de co-estabilización
Revisión v2.1: Valores HONESTOS · Streamlit 1.57 compatible · Arrow safe
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
VERSION = "v2.1 · Pergamino, Argentina · 2026"

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


# ── UTILIDAD: formateo seguro de floats ────────────────────
def fmt(v, decimals=8):
    """Formatea un valor numérico de forma segura, incluyendo NaN."""
    if v is None:
        return 'N/A'
    try:
        f = float(v)
        if np.isnan(f):
            return 'NaN'
        return f'{f:.{decimals}f}'
    except (TypeError, ValueError):
        return str(v)


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


# ── UTILIDADES VISUALES ──────────────────────────────────────
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
    d2_raw = result['metrics'].get('D2')
    d2 = float(d2_raw) if d2_raw is not None and not np.isnan(float(d2_raw)) else 0.0
    return h_spec, h_shan, d2


# ── FIGURAS ─────────────────────────────────────────────────
def fig_signal(x,label):
    fig,axes=plt.subplots(1,2,figsize=(10,2.8),facecolor=P['bg'])
    ax=axes[0];ax.set_facecolor(P['surface'])
    ax.plot(x[:600],lw=0.65,color=P['accent'],alpha=0.92)
    ax.fill_between(range(min(600,len(x))),x[:600],alpha=0.07,color=P['accent'])
    ax.set_xlabel('t',fontsize=8);ax.set_ylabel('x(t)',fontsize=8)
    ax.set_title(str(label)[:45],fontsize=8,color=P['accent']);ax.grid(True,alpha=0.25)
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
    ax.axhline(med,color=P['orange'],lw=1.2,ls='--',label=f'e~={med:.8f}')
    ax.set_xlabel('t (indice embedding)',fontsize=8);ax.set_ylabel('e(t)',fontsize=8)
    ax.set_title('epsilon dinamico - escala local del sistema',fontsize=9,color=P['accent'])
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
    ax.set_ylabel('y(t-tau)',fontsize=7,color=P['text'])
    ax.set_zlabel('y(t-2tau)',fontsize=7,color=P['text'])
    ax.set_title(f"tau={result['tau']} e={result['epsilon']:.8f}",color=P['accent'],fontsize=9,pad=8)
    for pane in [ax.xaxis.pane,ax.yaxis.pane,ax.zaxis.pane]:
        pane.fill=False;pane.set_edgecolor(P['border'])
    ax.tick_params(colors=P['border'],labelsize=5)
    sm=ScalarMappable(cmap=cmap,norm=norm);sm.set_array([])
    cb=fig.colorbar(sm,ax=ax,shrink=0.45,pad=0.1)
    cb.ax.tick_params(labelsize=5,colors=P['text']);cb.set_label('e(t)',color=P['text'],fontsize=6)
    fig.tight_layout();return fig

def fig_metrics(result):
    r3=result['R3'];sm=r3['stability_map'];delta=r3['delta']
    # 5 metricas si SampEn esta disponible
    all_keys=['lambda','D2','LZ','TE','SampEn']
    key_labels={'lambda':'lambda (Lyapunov)','D2':'D2 (Dim. Corr.)',
                'LZ':'C_LZ (Compl.)','TE':'TE (Trans. Ent.)','SampEn':'SampEn (Muestra)'}
    present_keys=[k for k in all_keys if k in sm]
    labels=[key_labels[k] for k in present_keys]
    grads=[sm[k].get('gradient',0.0) for k in present_keys]
    stable=[sm[k].get('stable',False) for k in present_keys]
    colors=[P['green'] if s else P['red'] for s in stable]

    fig,axes=plt.subplots(1,2,figsize=(11,4.0),facecolor=P['bg'])
    ax=axes[0];ax.set_facecolor(P['surface'])
    bars=ax.barh(labels,grads,color=colors,alpha=0.85,height=0.52)
    ax.axvline(delta,color=P['orange'],lw=1.8,ls='--',label=f'delta={delta} ({r3["regime"]})')
    ax.set_xlabel('|dmu/dtau| / RMS(mu) — gradiente normalizado',fontsize=7)
    ax.set_title('Sensibilidad a tau por metrica',fontsize=9,color=P['accent'])
    ax.legend(fontsize=7,framealpha=0.15);ax.grid(True,axis='x',alpha=0.25)
    mx=max(grads) if grads and max(grads)>0 else 0.01
    for bar,g in zip(bars,grads):
        ax.text(g+mx*0.03,bar.get_y()+bar.get_height()/2,f'{g:.7f}',
                va='center',ha='left',fontsize=6,color=P['text'])

    ax2=axes[1];ax2.set_facecolor(P['surface'])
    theta=np.linspace(np.pi,0,300)
    for i in range(len(theta)-1):
        c=plt.cm.RdYlGn(i/(len(theta)-1))
        ax2.fill_between([np.cos(theta[i]),np.cos(theta[i+1])],
                         [np.sin(theta[i])*0.68,np.sin(theta[i+1])*0.68],
                         [np.sin(theta[i])*1.0,np.sin(theta[i+1])*1.0],color=c,alpha=0.88)
    score=float(r3['R3_score']);angle=np.pi*(1-min(max(score,0),1))
    ax2.annotate('',xy=(0.83*np.cos(angle),0.83*np.sin(angle)),xytext=(0,0),
                 arrowprops=dict(arrowstyle='->',color=P['text'],lw=2.8),zorder=5)
    ax2.plot(0,0,'o',color=P['text'],ms=6,zorder=6)
    coh_c=P['green'] if r3['coherent'] else P['red']
    coh_t='[COHERENTE]' if r3['coherent'] else '[NO COHERENTE]'
    ax2.text(0,-0.20,f"R3 = {score:.8f}",ha='center',fontsize=13,fontweight='bold',color=P['accent'])
    ax2.text(0,-0.44,coh_t,ha='center',fontsize=10,color=coh_c)
    ax2.text(0,-0.64,r3['regime_desc'],ha='center',fontsize=8,color=P['orange'])
    ax2.text(-1.04,-0.1,'0',ha='center',fontsize=8,color=P['red'])
    ax2.text(1.04,-0.1,'1',ha='center',fontsize=8,color=P['green'])
    ax2.text(0,1.07,'0.5',ha='center',fontsize=8,color=P['orange'])
    ax2.set_xlim(-1.28,1.28);ax2.set_ylim(-0.88,1.22)
    ax2.set_aspect('equal');ax2.axis('off')
    ax2.set_title('R3 Score - co-estabilizacion (CONTINUO)',fontsize=9,color=P['accent'])
    fig.tight_layout(pad=1.5);return fig

def fig_baselines(x,result):
    h_spec,h_shan,d2=compute_baselines(x,result)
    r3_score=float(result['R3']['R3_score'])
    coherent=result['R3']['coherent']
    fig,ax=plt.subplots(figsize=(8,3.2),facecolor=P['bg'])
    ax.set_facecolor(P['surface'])
    methods=['H Fourier','H Shannon','D2 (norm)','R3']
    vals=[h_spec,h_shan,min(d2/3.0,1.0),r3_score]
    bar_c=[P['muted'],P['muted'],P['muted'],P['green'] if coherent else P['red']]
    bars=ax.bar(methods,vals,color=bar_c,alpha=0.85,width=0.55)
    ax.axhline(0.60,color=P['orange'],lw=1.2,ls='--',alpha=0.7,label='Umbral R3=0.60')
    ax.set_ylim(0,1.15);ax.set_ylabel('Valor normalizado',fontsize=8)
    ax.set_title('Comparacion: baselines vs R3',fontsize=9,color=P['accent'])
    ax.legend(fontsize=7,framealpha=0.15);ax.grid(True,axis='y',alpha=0.25)
    for bar,v in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2,v+0.02,f'{v:.7f}',
                ha='center',fontsize=7,color=P['text'])
    fig.tight_layout();return fig

def fig_windowed(sig,win_results):
    if not win_results:return None
    t_v=[r['t_s'] for r in win_results]
    r3_v=[float(r['R3']) for r in win_results]
    coh_v=[bool(r['coherente']) for r in win_results]
    tau_v=[int(r['tau']) for r in win_results]
    fig,axes=plt.subplots(3,1,figsize=(11,7),facecolor=P['bg'])
    fig.suptitle('Analisis por ventanas temporales',color=P['accent'],fontsize=11,fontweight='bold')
    ax1=axes[0];ax1.set_facecolor(P['surface'])
    ax1.plot(sig[:min(3000,len(sig))],lw=0.5,color=P['accent'],alpha=0.9)
    ax1.set_ylabel('Amplitud',fontsize=8);ax1.set_title('Senal',fontsize=8,color=P['text']);ax1.grid(True,alpha=0.2)
    ax2=axes[1];ax2.set_facecolor(P['surface'])
    bc=[P['green'] if c else P['red'] for c in coh_v]
    w=max((t_v[1]-t_v[0])*0.8 if len(t_v)>1 else 1,0.5)
    ax2.bar(t_v,r3_v,width=w,color=bc,alpha=0.85)
    ax2.axhline(0.60,color=P['orange'],lw=1.2,ls='--',alpha=0.8)
    ax2.set_ylim(0,1.15);ax2.set_ylabel('R3',fontsize=8)
    ax2.set_title('R3 por ventana (verde=coherente  rojo=incoherente)',fontsize=8,color=P['text']);ax2.grid(True,alpha=0.2)
    ax3=axes[2];ax3.set_facecolor(P['surface'])
    ax3.plot(t_v,tau_v,color=P['purple'],lw=1.2,marker='o',ms=3,alpha=0.9)
    ax3.fill_between(t_v,tau_v,alpha=0.1,color=P['purple'])
    ax3.set_xlabel('t (indice)',fontsize=8);ax3.set_ylabel('tau',fontsize=8)
    ax3.set_title('Evolucion de tau semidinamico',fontsize=8,color=P['text']);ax3.grid(True,alpha=0.2)
    fig.tight_layout(rect=[0,0,1,0.95]);return fig


# ── LECTOR MIT-BIH ──────────────────────────────────────────
def read_mitbih_bytes(dat_file, hea_file):
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


# ══════════════════════════════════════════════════════════════
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
        font-weight:600;letter-spacing:.05em;}}
    h1,h2,h3{{font-family:'Space Grotesk',sans-serif;color:{P['accent']};}}
    .stTabs [data-baseweb="tab"]{{font-family:'JetBrains Mono',monospace;color:{P['muted']};}}
    .stTabs [aria-selected="true"]{{color:{P['accent']} !important;border-bottom:2px solid {P['accent']} !important;}}
    </style>""",unsafe_allow_html=True)

    # HEADER
    col_t,col_i=st.columns([3,1])
    with col_t:
        st.markdown("# MODE Pipeline")
        st.markdown("**H1** epsilon dinamico  **H2** tau semidinamico  **H3** R3 co-estabilizacion")
    with col_i:
        st.markdown(
            f"<div style='text-align:right;padding-top:12px;"
            f"color:{P['muted']};font-family:JetBrains Mono,monospace;font-size:.74rem;'>"
            f"{AUTHOR}<br>{VERSION}</div>",
            unsafe_allow_html=True
        )
    st.divider()

    # SIDEBAR
    with st.sidebar:
        st.markdown("### Configuracion")
        st.divider()
        modo=st.radio("Modo",[
            "Senal sintetica",
            "Cargar CSV",
            "ECG MIT-BIH",
            "Ventanas temporales"
        ])
        st.divider()
        st.markdown("**Pipeline**")
        m_dim=st.slider("Dimension embedding m",2,6,3)
        max_tau=st.slider("tau maximo (AMI)",10,80,40,5)

        # Inicializar con array vacío (no dict)
        x_data = np.array([])
        label  = 'sin datos'
        extra  = {}

        if modo=="Senal sintetica":
            st.divider(); st.markdown("**Senal**")
            demo_name=st.selectbox("Seleccionar",list(DEMOS.keys()))
            N=st.slider("N muestras",300,3000,1000,100)
            x_data=DEMOS[demo_name](N); label=demo_name

        elif modo=="Cargar CSV":
            st.divider(); st.markdown("**Archivo**")
            up_csv=st.file_uploader("CSV (una columna)",type=['csv','txt'])
            if up_csv is not None:
                x_data=pd.read_csv(up_csv,header=None).iloc[:,0].dropna().values.astype(float)
                label=up_csv.name

        elif modo=="ECG MIT-BIH":
            st.divider(); st.markdown("**Archivos ECG**")
            st.caption("Subir el .dat y el .hea del mismo registro")
            up_dat=st.file_uploader("Archivo .dat",type=['dat'])
            up_hea=st.file_uploader("Archivo .hea",type=['hea'])
            ecg_start=st.slider("Inicio (s)",0,600,0,5)
            ecg_dur=st.slider("Duracion (s)",5,60,10,5)
            if up_dat is not None and up_hea is not None:
                try:
                    sig_full,fs_ecg,_=read_mitbih_bytes(up_dat,up_hea)
                    x_data=sig_full[int(ecg_start*fs_ecg):int((ecg_start+ecg_dur)*fs_ecg)]
                    label=f"ECG t={ecg_start}-{ecg_start+ecg_dur}s"
                    extra={'fs':fs_ecg}
                except Exception as e:
                    st.error(f"Error leyendo ECG: {e}")

        elif modo=="Ventanas temporales":
            st.divider(); st.markdown("**Fuente**")
            w_src=st.radio("",["Demo","CSV"],key='wsrc')
            if w_src=="Demo":
                w_demo=st.selectbox("Senal",list(DEMOS.keys()),key='wdm')
                w_N=st.slider("N total",1000,5000,2000,100,key='wN')
                x_data=DEMOS[w_demo](w_N); label=w_demo
            else:
                w_csv=st.file_uploader("CSV",type=['csv','txt'],key='wcsv')
                if w_csv is not None:
                    x_data=pd.read_csv(w_csv,header=None).iloc[:,0].dropna().values.astype(float)
                    label=w_csv.name
            st.divider(); st.markdown("**Ventanas**")
            win_size=st.slider("Tamano (muestras)",200,2000,1000,100)
            win_step=st.slider("Paso (muestras)",100,1000,500,50)
            win_max=st.slider("Max. ventanas",5,50,20,5)
            extra={'win_size':win_size,'win_step':win_step,'win_max':win_max}

        st.divider()
        run_btn=st.button("Ejecutar pipeline",type="primary",use_container_width=True)
        st.divider()
        st.markdown(
            f"<div style='font-family:JetBrains Mono,monospace;font-size:.72rem;color:{P['muted']};'>"
            f"<b style='color:{P['text']}'>delta por regimen</b><br><br>"
            f"Estable &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0.06<br>"
            f"Caos debil &nbsp;&nbsp; 0.05<br>"
            f"Caotico &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0.08<br>"
            f"Hipercaotico &nbsp;0.15<br>"
            f"Ruidoso &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0.20<br><br>"
            f"<b style='color:{P['text']}'>Umbral coherencia</b><br>"
            f"R3 mayor= 0.60 (continuo)</div>",
            unsafe_allow_html=True
        )

    # ── EJECUCION ───────────────────────────────────────────
    if run_btn:
        if len(x_data) == 0:
            st.warning("Cargar datos primero."); st.stop()

        if modo == "Ventanas temporales":
            ws=extra.get('win_size',1000)
            wst=extra.get('win_step',500)
            wmax=extra.get('win_max',20)
            with st.spinner("Analizando ventanas..."):
                pipe=AttractorPipeline(m=m_dim,max_tau=max_tau,verbose=False)
                win_rs=[]; prog=st.progress(0); nw=0
                for start in range(0,len(x_data)-ws,wst):
                    if nw>=wmax: break
                    w=x_data[start:start+ws]
                    if w.std()<1e-6: continue
                    try:
                        r=pipe.run(w,label=f"w{nw}"); r3=r['R3']
                        win_rs.append({
                            't_s':    int(start),
                            'tau':    int(r['tau']),
                            'epsilon':float(r['epsilon']),
                            'R3':     float(r3['R3_score']),
                            'coherente': bool(r3['coherent']),
                            'regimen':   str(r3['regime']),
                            'delta':     float(r3['delta']),
                        })
                    except Exception as e:
                        pass
                    nw+=1; prog.progress(min(nw/wmax,1.0))
                prog.empty()
            st.session_state.update({'mode':modo,'x':x_data,'label':label,
                                     'win_results':win_rs,'result':None})
        else:
            with st.spinner("Calculando epsilon, tau, metricas, R3..."):
                try:
                    pipe=AttractorPipeline(m=m_dim,max_tau=max_tau,verbose=False)
                    r=pipe.run(x_data,label=label)
                    st.session_state.update({'mode':modo,'x':x_data,'label':label,
                                             'result':r,'win_results':None})
                except Exception as e:
                    st.error(f"Error en pipeline: {e}")
                    st.code(traceback.format_exc()); st.stop()

    # ── PANTALLA INICIAL ─────────────────────────────────────
    if 'mode' not in st.session_state:
        st.markdown(
            f"<div style='text-align:center;padding:60px 20px;'>"
            f"<div style='font-size:3.5rem;margin-bottom:16px;'>🌀</div>"
            f"<h2 style='color:{P['accent']};font-family:JetBrains Mono,monospace;'>MODE Pipeline v2.1</h2>"
            f"<p style='color:{P['muted']};font-size:.95rem;max-width:680px;margin:0 auto;line-height:1.7;'>"
            f"Framework de legibilidad observacional para sistemas dinamicos no lineales.<br>"
            f"<b>Revision 2026-05:</b> Calculos HONESTOS · Gradientes RMS · R3 continuo · Arrow-safe<br>"
            f"Seleccionar un modo en el sidebar y presionar Ejecutar pipeline."
            f"</p><br>"
            f"<p style='color:{P['border']};font-family:JetBrains Mono,monospace;font-size:.75rem;'>"
            f"{AUTHOR} &nbsp;&middot;&nbsp; {VERSION}</p></div>",
            unsafe_allow_html=True
        )
        st.stop()

    modo_act=st.session_state.get('mode','')

    # ── MODO VENTANAS ────────────────────────────────────────
    if modo_act=="Ventanas temporales":
        win_rs=st.session_state.get('win_results',[])
        sig_w=st.session_state.get('x',np.array([]))
        if not win_rs:
            st.info("Presionar Ejecutar pipeline para analizar."); st.stop()
        coh_n=sum(1 for r in win_rs if r['coherente'])
        r3_med=np.mean([r['R3'] for r in win_rs])
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Ventanas",len(win_rs))
        c2.metric("R3 promedio",f"{r3_med:.8f}")
        c3.metric("Coherentes",f"{coh_n} ({coh_n/len(win_rs)*100:.1f}%)")
        c4.metric("Senal",str(st.session_state.get('label',''))[:25])
        st.divider()
        fw=fig_windowed(sig_w,win_rs)
        if fw: st.image(to_png(fw),width='stretch')
        st.divider()
        st.subheader("Resultados por ventana")
        # Asegurar tipos consistentes para Arrow
        df_w=pd.DataFrame(win_rs)
        for col in df_w.columns:
            if df_w[col].dtype == object:
                df_w[col] = df_w[col].astype(str)
        st.dataframe(df_w,use_container_width=True,hide_index=True)
        st.download_button("Descargar CSV ventanas",
                           df_w.to_csv(index=False).encode(),
                           "ventanas.csv",mime="text/csv")
        st.stop()

    # ── MODOS NORMALES ───────────────────────────────────────
    result=st.session_state.get('result')
    x=st.session_state.get('x',np.array([]))
    label=st.session_state.get('label','')
    if result is None:
        st.info("Presionar Ejecutar pipeline"); st.stop()

    r3=result['R3']; mvals=result['metrics']
    r3_score = float(r3['R3_score'])

    # Metricas rapidas
    c1,c2,c3,c4,c5,c6=st.columns(6)
    c1.metric("tau semidinamico", str(result['tau']))
    c2.metric("epsilon mediana",  f"{result['epsilon']:.8f}")
    c3.metric("R3 Score",         f"{r3_score:.8f}",
              delta="coherente" if r3['coherent'] else "no coherente")
    c4.metric("Regimen",  str(r3['regime']))
    c5.metric("delta activo", str(r3['delta']))
    c6.metric("N", str(len(x)))
    st.divider()

    tab1,tab2,tab3,tab4,tab5=st.tabs([
        "Senal & epsilon",
        "Atractor",
        "Metricas & R3",
        "Baselines",
        "Export"
    ])

    with tab1:
        st.subheader("Serie temporal y espectro")
        st.image(to_png(fig_signal(x,label)),width='stretch')
        st.subheader("epsilon(t) dinamico")
        st.image(to_png(fig_epsilon(result)),width='stretch')

    with tab2:
        fa=fig_attractor(result)
        if fa:
            col_a,col_b=st.columns([2,1])
            with col_a:
                st.subheader("Atractor reconstruido")
                st.image(to_png(fa),width='stretch')
            with col_b:
                st.markdown(
                    f"<div style='background:{P['card']};border:1px solid {P['border']};"
                    f"border-radius:10px;padding:20px;margin-top:40px;"
                    f"font-family:JetBrains Mono,monospace;font-size:.82rem;'>"
                    f"<div style='color:{P['muted']};margin-bottom:8px;'>EMBEDDING</div>"
                    f"tau = <b style='color:{P['accent']};'>{result['tau']}</b><br>"
                    f"m = <b style='color:{P['accent']};'>{result['embedding'].shape[1]}</b><br>"
                    f"e~ = <b style='color:{P['teal']};'>{result['epsilon']:.8f}</b><br><br>"
                    f"<div style='color:{P['muted']};margin-bottom:8px;'>REGIMEN</div>"
                    f"<div style='color:{P['orange']};'>{r3['regime_desc']}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
        else:
            st.info("Se necesita m mayor o igual a 3 para el atractor 3D.")

    with tab3:
        st.subheader("Sensibilidad a tau y R3 gauge")
        st.image(to_png(fig_metrics(result)),width='stretch')
        st.divider()
        st.subheader("Detalle por metrica (valores sin truncado)")
        sm_map=r3['stability_map']
        key_labels={'lambda':'lambda Lyapunov','D2':'D2 Corr.',
                    'LZ':'C_LZ Compl.','TE':'TE Transf.','SampEn':'SampEn Muestra'}
        cols_m=st.columns(min(len(sm_map),5))
        for i,(k,vd) in enumerate(sm_map.items()):
            with cols_m[i % len(cols_m)]:
                col=P['green'] if vd.get('stable',False) else P['red']
                grad_val = float(vd.get('gradient',0))
                weight_val = float(vd.get('weight',0))
                st.markdown(
                    f"<div style='background:{P['card']};border:1px solid {col}40;"
                    f"border-radius:10px;padding:14px;text-align:center;'>"
                    f"<div style='color:{P['muted']};font-size:.72rem;margin-bottom:6px;'>"
                    f"{key_labels.get(k,k)}</div>"
                    f"<div style='font-size:1.3rem;'>{'[OK]' if vd.get('stable',False) else '[--]'}</div>"
                    f"<div style='color:{P['text']};font-family:JetBrains Mono,monospace;font-size:.70rem;margin-top:6px;'>"
                    f"grad={grad_val:.8f}<br>"
                    f"weight={weight_val:.8f}<br>"
                    f"<span style='color:{P['muted']};'>delta={vd.get('delta',0)}</span>"
                    f"</div></div>",
                    unsafe_allow_html=True
                )

    with tab4:
        st.subheader("Comparacion con metodos estandar")
        st.image(to_png(fig_baselines(x,result)),width='stretch')
        st.divider()
        h_spec,h_shan,d2=compute_baselines(x,result)
        # FIX Arrow: todas las filas con tipos uniformes (str)
        bl_rows=[
            ('H Fourier espectral', fmt(h_spec,7), '1=plano',    'No distingue caos-ruido'),
            ('H Shannon (senal)',   fmt(h_shan,7), '1=max desor', 'Sin info temporal'),
            ('D2 clasico',         fmt(d2,7),      'Lorenz~2.05', 'Escalar puro'),
            ('R3 Score',           fmt(r3_score,8),'mayor=0.60->coherente','Continuo+regimen'),
            ('Coherente',          'SI' if r3['coherent'] else 'NO', '---','---'),
            ('Regimen',            str(r3['regime_desc']), '---', 'delta semidinamico'),
        ]
        df_bl=pd.DataFrame(bl_rows,columns=['Metodo','Valor','Ref','Nota'])
        df_bl=df_bl.astype(str)  # garantizar tipos uniformes
        st.dataframe(df_bl,use_container_width=True,hide_index=True)

    with tab5:
        st.subheader("Tabla completa — valores sin truncado ni redondeo")

        # FIX Arrow: todas las columnas como str explícitamente
        rows=[
            ('tau semidinamico',   str(result['tau']),                    '---'),
            ('tau inicial',        str(result.get('tau_initial','---')),  '---'),
            ('epsilon mediana',    fmt(result['epsilon'],8),              '---'),
            ('Regimen',            str(r3['regime_desc']),                '---'),
            ('delta activo',       str(r3['delta']),                     '---'),
            ('R3 Score',           fmt(r3_score,8),                       'mayor=0.60 continuo'),
            ('Coherente',          'SI' if r3['coherent'] else 'NO',      '---'),
            ('n_validas',          str(r3.get('n_valid','---')),          'metricas validas'),
            ('lambda (Lyapunov)',  fmt(mvals.get('lambda'),8),            'menor 0=estable, mayor 0=caos'),
            ('D2 (dim. corr.)',    fmt(mvals.get('D2'),8),                'Lorenz~2.05'),
            ('C_LZ (compl.)',      fmt(mvals.get('LZ'),8),                '0=orden, 1=compl'),
            ('TE (trans. ent.)',   fmt(mvals.get('TE'),8),                'Flujo inf'),
            ('SampEn (muestra)',   fmt(mvals.get('SampEn'),8),            'Richman & Moorman 2000'),
        ]
        df_res=pd.DataFrame(rows,columns=['Variable','Valor','Referencia'])
        df_res=df_res.astype(str)  # garantizar tipos uniformes para Arrow
        st.dataframe(df_res,use_container_width=True,hide_index=True)
        st.divider()

        ce1,ce2,ce3=st.columns(3)
        with ce1:
            st.download_button("Resultados CSV",
                               df_res.to_csv(index=False).encode(),
                               "resultados.csv",mime="text/csv")
        with ce2:
            emb=result['embedding']
            cols_=[f'y_t-{i*result["tau"]}' for i in range(emb.shape[1])]
            df_emb=pd.DataFrame(emb,columns=cols_)
            df_emb['epsilon']=result['epsilon_series'][:len(df_emb)]
            st.download_button("Embedding CSV",
                               df_emb.to_csv(index=False).encode(),
                               "embedding.csv",mime="text/csv")
        with ce3:
            h_spec2,h_shan2,d2_2=compute_baselines(x,result)
            df_bl2=pd.DataFrame({
                'metrica':['H_Fourier','H_Shannon','D2','R3','coherente','regimen'],
                'valor':  [fmt(h_spec2,8),fmt(h_shan2,8),fmt(d2_2,8),
                           fmt(r3_score,8),
                           str(r3['coherent']),str(r3['regime'])]
            })
            st.download_button("Baselines CSV",
                               df_bl2.to_csv(index=False).encode(),
                               "baselines.csv",mime="text/csv")

    st.markdown(
        f"<div style='text-align:center;padding:10px;margin-top:6px;"
        f"color:{P['border']};font-family:JetBrains Mono,monospace;font-size:.68rem;'>"
        f"{AUTHOR} &nbsp;&middot;&nbsp; {VERSION}<br>"
        f"Gradientes RMS normalizados &middot; R3 continuo ponderado &middot; Precision 8 decimales"
        f"</div>",
        unsafe_allow_html=True
    )

if __name__=='__main__':
    main()
