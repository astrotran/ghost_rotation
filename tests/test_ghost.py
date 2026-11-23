import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import GHOST_V0_2 as g


def lightcurve(period=2.8):
    t=np.arange(0,27,.03)
    y=1+.015*np.sin(2*np.pi*t/period)+np.random.default_rng(7).normal(0,.0004,len(t))
    return g.LightCurveData(t,y,np.full_like(t,.0004),name='synthetic')

@pytest.mark.parametrize('method',['LS','GLS','FFT','ACF','BGLS'])
def test_period_recovery(method):
    _,a=g.calculate_period_analysis(lightcurve(),g.FitSettings(pmin=.5,pmax=8,bgls_max_steps=1000),'all',method)
    assert len(a.measurements)==3
    for m in a.measurements.values():
        assert m.period==pytest.approx(2.8,rel=.035)
        assert m.peaks
        assert m.peaks[0]['rank']==1
        assert all(np.isfinite(p['period_days']) for p in m.peaks)


def test_ls_and_gls_are_different_models_and_weights_matter():
    lc=lightcurve();lc.flux[::10]+=.4*np.sin(2*np.pi*lc.time[::10]/1.4);lc.flux_err[::10]=1e3
    settings=g.FitSettings(pmin=.5,pmax=8,clip_upper=100,clip_lower=100)
    clean=g.preprocess_lightcurve(lc,settings)
    ls=g.calculate_classical_period(clean,settings,'LS');gls=g.calculate_classical_period(clean,settings,'GLS')
    assert not ls.diagnostic_metadata['fit_mean']
    assert gls.diagnostic_metadata['fit_mean']
    assert gls.period==pytest.approx(2.8,rel=.01)
    assert not np.allclose(ls.power,gls.power)


def test_harmonic_peak_and_manual_fold():
    lc=lightcurve(period=4.0)
    lc.flux=1+.03*np.sin(2*np.pi*lc.time/2)+.015*np.sin(2*np.pi*lc.time/4)
    clean,a=g.calculate_period_analysis(lc,g.FitSettings(pmin=.8,pmax=8),'full_sector','GLS')
    m=a.measurement('full_sector','GLS')
    assert m.period==pytest.approx(2,rel=.015)
    assert any(p['harmonic_candidate']=='near 2P' for p in m.peaks)
    folded=g.folded_dataframe(clean,4,epoch=clean.time[0])
    np.testing.assert_allclose(folded.phase_P,((clean.time-clean.time[0])/4)%1)
    assert np.all((folded.phase_P>=0)&(folded.phase_P<1))
    assert m.period<2.1  # Inspection does not silently adopt the alternative.


def test_fft_does_not_bridge_gap_and_rejects_irregular_sampling():
    lc=lightcurve();keep=(lc.time<11)|(lc.time>14)
    lc=g.LightCurveData(lc.time[keep],lc.flux[keep],lc.flux_err[keep])
    clean=g.preprocess_lightcurve(lc,g.FitSettings())
    m=g.calculate_classical_period(clean,g.FitSettings(pmin=.5,pmax=8),'FFT')
    assert m.diagnostic_metadata['run_points']<len(clean.time)
    assert m.diagnostic_metadata['run_start']>14
    assert m.period==pytest.approx(2.8,rel=.07)
    t=np.arange(100)*.1+np.where(np.arange(100)%2,.02,0)
    m=g.calculate_classical_period(g.LightCurveData(t,np.sin(t),np.ones(100)),g.FitSettings(pmax=5),'FFT')
    assert not np.isfinite(m.period)
    assert 'irregular' in m.note

@pytest.mark.parametrize('relation,kwargs,expected',[
 ('Noyes1984',{'b_v':.65},11.987783059367715),
 ('Wright2011',{'v_ks':2.},14.791083881682072),
 ('Wright2011',{'v_ks':5.},123.02687708123811),
 ('Cranmer2011',{'teff':5772.},12.875198249062509)])
def test_rossby_reference_values(relation,kwargs,expected):
    tau,*_=g.turnover_time(relation,**kwargs)
    assert tau==pytest.approx(expected,rel=1e-10)
    rr=g.calculate_rossby_number(b_v=kwargs.get('b_v',np.nan),period_source='manual',analysis=None,period_override=5,
        relation=relation,v_ks=kwargs.get('v_ks',np.nan),teff=kwargs.get('teff',np.nan))
    assert rr.rossby_number==pytest.approx(5/tau)
    assert rr.relation==relation

@pytest.mark.parametrize('relation,kwargs',[('Noyes1984',{'b_v':.1}),('Wright2011',{'v_ks':8}),('Cranmer2011',{'teff':8000})])
def test_rossby_extrapolation_is_explicit(relation,kwargs):
    with pytest.raises(ValueError,match='outside'):g.turnover_time(relation,**kwargs)
    assert not g.turnover_time(relation,allow_extrapolation=True,**kwargs)[3]


def test_exact_catalog_routing_and_ks_provenance(monkeypatch):
    queries=[]
    def tap(endpoint,query,timeout=25):
        queries.append(query)
        if 'SELECT DISTINCT' in query:return [{'oid':'42','main_id':'Test Star'}]
        if 'FROM allfluxes' in query:return [{'b':'10.65','v':'10.0','k':'0.0'}]
        if "LIKE '2MASS" in query:return [{'id':'2MASS J12345678+1234567'}]
        if 'FROM fp_psc' in query:return [{'designation':'12345678+1234567','ks_m':'8.0','ks_cmsig':'.02','ph_qual':'AAA'}]
        if 'FROM mesFe_h' in query:return [{'teff':'5000','bibcode':'ref1'},{'teff':'5200','bibcode':'ref2'}]
        raise AssertionError(query)
    monkeypatch.setattr(g,'_tap_rows',tap)
    w=g.query_rossby_inputs('Test','Wright2011')
    assert w['v_ks']==2 # Never uses the unverified generic K value.
    assert w['ks_mag']==8
    assert 'exact designation' in w['input_provenance']['ks_source']
    c=g.query_rossby_inputs('Test','Cranmer2011')
    assert c['teff']==5100
    assert len(c['input_provenance']['teff_measurements'])==2
    n=g.query_rossby_inputs('Test','Noyes1984')
    assert n['b_v']==pytest.approx(.65)


def test_missing_or_low_quality_ks_does_not_fall_back_to_k(monkeypatch):
    def tap(endpoint,query,timeout=25):
        if 'SELECT DISTINCT' in query:return [{'oid':'42','main_id':'Test Star'}]
        if 'FROM allfluxes' in query:return [{'b':'10.65','v':'10.0','k':'8.0'}]
        if "LIKE '2MASS" in query:return [{'id':'2MASS J12345678+1234567'}]
        return [{'designation':'12345678+1234567','ks_m':'8','ph_qual':'AAU'}]
    monkeypatch.setattr(g,'_tap_rows',tap)
    with pytest.raises(ValueError,match='quality'):g.query_rossby_inputs('Test','Wright2011')


def fake_analysis(values):
    measurements={(seg,'LS'):g.PeriodMeasurement(seg,'LS',v) for seg,v in zip(g.SEGMENT_ORDER,values)}
    return g.PeriodAnalysis(measurements,np.nan,np.nan,float(np.median(values)),0,0,list(g.SEGMENT_ORDER),['LS'])


def test_sector_then_star_aggregation_and_unknown_sector():
    records=[{'path':f'A_{i}.csv','metadata':g.LCMetadata('A','1','QLP'),'analysis':fake_analysis([1,1,1])} for i in range(4)]
    records += [{'path':'A_5.csv','metadata':g.LCMetadata('A','2','QLP'),'analysis':fake_analysis([9,9,9])},
                {'path':'A_unknown.csv','metadata':g.LCMetadata('A','','QLP'),'analysis':fake_analysis([100,100,100])}]
    detail,lc,stars=g.build_batch_tables(records,'final_averaged_prot','median')
    assert set(detail.note)=={'LS'}
    assert stars.iloc[0].final_median_period==5
    assert stars.iloc[0].n_sectors_used==2
    assert 'ls_full_sector' in lc


def test_plotting_and_strict_json_without_gp(tmp_path):
    from matplotlib.figure import Figure
    clean,a=g.calculate_period_analysis(lightcurve(),g.FitSettings(pmax=8),'full_sector','GLS')
    fig=Figure();g.draw_period_analysis_figure(fig,clean,a);fig.savefig(tmp_path/'overview.png')
    fig=Figure();g.draw_peak_phase_figure(fig,clean,a.measurement('full_sector','GLS'));fig.savefig(tmp_path/'phase.png')
    payload=g.strict_json(g.analysis_json(clean,a,g.FitSettings()))
    json.dumps(payload,allow_nan=False)
    assert payload['measurements'][0]['peaks']


def test_cli_no_gp_and_rossby_exports(tmp_path):
    lc=lightcurve();source=tmp_path/'synthetic.csv'
    pd.DataFrame({'time':lc.time,'flux':lc.flux,'flux_err':lc.flux_err}).to_csv(source,index=False)
    args=g.build_arg_parser().parse_args(['--file',str(source),'--period-method','GLS','--segment','full_sector','--pmax','8',
       '--calculate-rossby','--rossby-relation','Wright2011','--v-ks','2','--output',str(tmp_path/'result')])
    assert not args.run_gp
    assert g.run_cli(args)==0
    payload=json.loads((tmp_path/'result.json').read_text())
    assert payload['rossby']['relation']=='Wright2011'
    assert (tmp_path/'result_phase.png').exists()
    assert (tmp_path/'result_folded.csv').exists()
    assert (tmp_path/'result_peaks.csv').exists()


def test_rossby_propagation_before_photometry_is_available():
    records = [{'path':'A.csv','metadata':g.LCMetadata('A','1','QLP'),
                'analysis':fake_analysis([2,3,4])}]
    detail, lc, stars = g.build_batch_tables(records, 'final_averaged_prot', 'median')
    detail, lc = g.propagate_star_rossby_to_batch_tables(detail, lc, stars)
    assert lc.rossby_number.isna().all()
    assert detail.rossby_relation.isna().all()
