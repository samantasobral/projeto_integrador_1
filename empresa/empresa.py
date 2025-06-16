import pickle
import pandas as pd
import numpy as np
import json

class PredictEmprestimo(object):
    def __init__(self):
        self.idade                              = pickle.load(open('parameter/idade_robust.pkl', 'rb'))
        self.renda                              = pickle.load(open('parameter/renda_robust.pkl', 'rb'))
        self.tempo_emprego                      = pickle.load(open('parameter/tempo_emprego_robust.pkl', 'rb'))
        self.relacao_emprestimo_renda           = pickle.load(open('parameter/relacao_emprestimo_renda_robust.pkl', 'rb'))
        self.tx_juros_pela_renda                = pickle.load(open('parameter/tx_juros_pela_renda_robust.pkl', 'rb'))
        self.valor_emprestimo_pela_idade        = pickle.load(open('parameter/valor_emprestimo_pela_idade_robust.pkl', 'rb'))
        self.valor_emprestimo_pelo_tempo        = pickle.load(open('parameter/valor_emprestimo_pelo_tempo_robust.pkl', 'rb'))
        self.valor_emprestimo_pelo_historico    = pickle.load(open('parameter/valor_emprestimo_pelo_historico_robust.pkl', 'rb'))
        self.renda_pelo_valor_emprestimo        = pickle.load(open('parameter/renda_pelo_valor_emprestimo_robust.pkl', 'rb'))
        self.valor_emprestimo                   = pickle.load(open('parameter/valor_emprestimo_minmax.pkl', 'rb'))
        self.taxa_juros_emprestimo              = pickle.load(open('parameter/taxa_juros_emprestimo_minmax.pkl', 'rb'))
        self.historico_credito                  = pickle.load(open('parameter/historico_credito_minmax.pkl', 'rb'))
        self.finalidade_emprestimo              = pickle.load(open('parameter/finalidade_emprestimo_scaler.pkl', 'rb'))
        self.grau_risco_emprestimo              = pickle.load(open('parameter/grau_risco_emprestimo_scaler.pkl', 'rb'))
        self.posse_casa                         = pickle.load(open('parameter/posse_casa_scaler.pkl', 'rb'))
        self.registro_inadimplencia             = pickle.load(open('parameter/registro_inadimplencia_scaler.pkl', 'rb'))
        self.imputer                            = pickle.load(open('parameter/imputer_knn.pkl', 'rb'))

    def data_cleaning(self, df1):
        num_attributes = df1.select_dtypes(include = ['int64', 'float64'])
        cat_attributes = df1.select_dtypes(include = ['object', 'category'])
        if num_attributes.isna().any().any():
            df_imputed = pd.DataFrame(self.imputer.transform(num_attributes), columns = num_attributes.columns)
            df1 = pd.concat([df_imputed, cat_attributes], axis = 1)
        else:
            return df1
        return df1

    def feature_engineering(self, df2):
        # taxa de juros ajustada a renda (proporção entre a tx de juros e a renda):
        df2['tx_juros_pela_renda'] = df2['taxa_juros_emprestimo'] / df2['renda']

        #proporção do valor do empréstimo em relação a idade:
        df2['valor_emprestimo_pela_idade'] = df2['valor_emprestimo'] / df2['idade']

        #proporção do valor do empréstimo em relação ao tempo de emprego:
        df2['tempo_emprego'] = np.where(df2['tempo_emprego'] == 0, 0.5, df2['tempo_emprego'])
        df2['valor_emprestimo_pelo_tempo'] = df2['valor_emprestimo'] / df2['tempo_emprego']

         #proporção da renda pelo valor do empréstimo:
        df2['renda_pelo_valor_emprestimo'] = df2['renda'] / df2['valor_emprestimo']

         #proporção do  valor do empréstimo pelo histórico de crédito:
        df2['valor_emprestimo_pelo_historico'] = df2['valor_emprestimo'] / df2['historico_credito']

        return df2

    def data_preparation(self, df3):
        df3['idade'] = self.idade.transform(df3[['idade']])
        df3['renda'] = self.renda.transform(df3[['renda']])
        df3['tempo_emprego'] = self.tempo_emprego.transform(df3[['tempo_emprego']])
        df3['relacao_emprestimo_renda'] = self.relacao_emprestimo_renda.transform(df3[['relacao_emprestimo_renda']])
        df3['tx_juros_pela_renda'] = self.tx_juros_pela_renda.transform(df3[['tx_juros_pela_renda']])
        df3['valor_emprestimo_pela_idade'] = self.valor_emprestimo_pela_idade.transform(df3[['valor_emprestimo_pela_idade']])
        df3['valor_emprestimo_pelo_tempo'] = self.valor_emprestimo_pelo_tempo.transform(df3[['valor_emprestimo_pelo_tempo']])
        df3['valor_emprestimo_pelo_historico'] = self.valor_emprestimo_pelo_historico.transform(df3[['valor_emprestimo_pelo_historico']])
        df3['renda_pelo_valor_emprestimo'] = self.renda_pelo_valor_emprestimo.transform(df3[['renda_pelo_valor_emprestimo']])
        df3['valor_emprestimo'] = self.valor_emprestimo.transform(df3[['valor_emprestimo']])
        df3['taxa_juros_emprestimo'] = self.taxa_juros_emprestimo.transform(df3[['taxa_juros_emprestimo']])
        df3['historico_credito'] = self.historico_credito.transform(df3[['historico_credito']])
        df3['finalidade_emprestimo'] = self.finalidade_emprestimo.transform(df3['finalidade_emprestimo'])
        df3['grau_risco_emprestimo'] = self.grau_risco_emprestimo.transform(df3['grau_risco_emprestimo'])
        df3['posse_casa'] = self.posse_casa.transform(df3['posse_casa'])
        df3['registro_inadimplencia'] = self.registro_inadimplencia.transform(df3['registro_inadimplencia'])

        boruta_columns = ['renda',
                        'valor_emprestimo',
                        'taxa_juros_emprestimo',
                        'relacao_emprestimo_renda',
                        'posse_casa',
                        'finalidade_emprestimo',
                        'grau_risco_emprestimo',
                        'tx_juros_pela_renda',
                        'valor_emprestimo_pela_idade',
                        'valor_emprestimo_pelo_tempo',
                        'renda_pelo_valor_emprestimo']

        return df3[boruta_columns] 

    def get_predictions(self, model, test_data, original_data):
        pred = model.predict(test_data)
        original_data['prediction'] = pred

        return original_data.to_json(orient = 'records')
