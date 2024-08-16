"""
    Code to extract some examples of where the attention was focusing for input documents
"""
import operator
import random
import re
import sys
from copy import deepcopy

import learn.models as models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


def save_attn_colored(data, output, target_data, s, filter_size, attn_colored_dir, latex_template, hadm_id,
                      df_raw_data, df_spans, dicts=None):
    output_rd = np.round(output)
    pred_codes = np.where(output_rd[0] == 1)[0]
    tgt_codes = np.where(target_data[0] == 1)[0]
    if dicts is not None:
        if s is not None and len(pred_codes) > 0:
            if df_raw_data is not None and df_spans is not None:
                important_spans_colored_raw(data, output, tgt_codes, pred_codes, s, dicts, filter_size,
                                            attn_colored_dir, latex_template,
                                            df_raw_data, df_spans, hadm_id)
            else:
                important_spans_colored(data, output, tgt_codes, pred_codes, s, dicts, filter_size, attn_colored_dir,
                                        latex_template, hadm_id)


def important_spans_colored_bak(data, output, tgt_codes, pred_codes, s, dicts, filter_size, attn_colored_dir,
                                latex_template, hadm_id):
    tex_file = open('%s/%s.tex' % (attn_colored_dir, hadm_id), 'w')
    ind2w, ind2c, desc_dict = dicts['ind2w'], dicts['ind2c'], dicts['desc']

    # Make a list with lists containing the words of the document (later attn score will be added)
    doc_attn = [[ind2w[w], 0] if w in ind2w.keys() else ['UNK', 0] for w in data[0].data.cpu().numpy()]
    doc_len = len(doc_attn)

    tgt_codes_string = "\\textbf{REAL CODES:}\\newline\n"
    for tgt_code in tgt_codes:
        tgt_icd_code = ind2c[tgt_code]
        tgt_icd_code_desc = desc_dict[tgt_icd_code]
        tgt_codes_string += "%i: %s, %s\\newline\n" % \
                            (tgt_code, tgt_icd_code, tgt_icd_code_desc)

    p_codes_string = "\\textbf{PREDICTED CODES:}\\newline\n"

    docs_with_spans = []
    for p_code in pred_codes:
        tp = p_code in tgt_codes
        p_icd_code = ind2c[p_code]
        p_icd_code_desc = desc_dict[p_icd_code]
        conf = output[0][p_code]
        p_codes_string += "%i [%s]: %s, %s, conf: %f\\newline\n" % \
                          (p_code, "TP" if tp else "FP", p_icd_code, p_icd_code_desc, conf)

        # Get current predicted code's attention vector
        attn = s[0][p_code].data.cpu().numpy()

        # Copy doc for the current iteration
        current_doc_attn = deepcopy(doc_attn)

        # Add attention scores to current doc
        for start in range(doc_len):
            end = start + filter_size
            for i in range(start, end):
                if end <= doc_len:
                    current_doc_attn[i][1] += attn[i]

        # Scale attention scores and create LaTeX colorboxes
        current_p_code_latex = '\\textbf{Spans for code %i:}\\newline' % p_code
        for i in range(doc_len):
            # Attention scores are scaled to the interval [0, 100] as LaTeX xcolor
            # package's intesity uses this interval
            word = current_doc_attn[i][0]
            attn_score = int(((current_doc_attn[i][1] / filter_size)) * 10 ** 2)

            current_p_code_latex += '\colorbox{blue!%s}{%s} ' % (attn_score, word)

        docs_with_spans.append(current_p_code_latex + '\\newline\n')

    # Concatenate every part to create the full LaTeX output
    full_output = tgt_codes_string + '\\newline\n' + p_codes_string + '\\newline\n'

    for doc in docs_with_spans:
        full_output += doc + '\n'

    tex_file.write(latex_template.replace('YourTextHere', full_output))
    tex_file.close()


def important_spans_colored(data, output, tgt_codes, pred_codes, s, dicts, filter_size, attn_colored_dir,
                            latex_template, hadm_id):
    tex_file = open('%s/%s.tex' % (attn_colored_dir, hadm_id), 'w')
    ind2w, ind2c, desc_dict = dicts['ind2w'], dicts['ind2c'], dicts['desc']

    # Make a list with lists containing the words of the document (later attn score will be added)
    doc_attn = [[ind2w[w], 0] if w in ind2w.keys() else ['UNK', 0] for w in data[0].data.cpu().numpy()]
    doc_len = len(doc_attn)

    tgt_codes_string = "\\textbf{REAL CODES:}\\newline\n"
    for tgt_code in tgt_codes:
        tgt_icd_code = ind2c[tgt_code]
        tgt_icd_code_desc = desc_dict[tgt_icd_code]
        tgt_codes_string += "%i: %s, %s\\newline\n" % \
                            (tgt_code, tgt_icd_code, tgt_icd_code_desc)

    p_codes_string = "\\textbf{PREDICTED CODES:}\\newline\n"

    docs_with_spans = []
    for p_code in pred_codes:
        tp = p_code in tgt_codes
        p_icd_code = ind2c[p_code]
        p_icd_code_desc = desc_dict[p_icd_code]
        conf = output[0][p_code]
        p_codes_string += "%i [%s]: %s, %s, conf: %f\\newline\n" % \
                          (p_code, "TP" if tp else "FP", p_icd_code, p_icd_code_desc, conf)

        # Get current predicted code's attention vector
        attn = s[0][p_code].data.cpu().numpy()
        # Takes indices of the 10 greatest attention scores
        imps = attn.argsort()[-10:][::-1]
        windows = make_windows(imps, filter_size, attn)

        # Copy doc for the current iteration
        current_doc_attn = deepcopy(doc_attn)

        # Mark words to be colorboxed
        colorboxed = [0] * doc_len
        window_num = 0
        for (start, end), score in windows:
            for i in range(start, end):
                try:
                    colorboxed[i] = window_num
                except:
                    pass
            window_num += 1

        current_p_code_latex = '\\textbf{Spans for code %i:}\\newline\n' % p_code
        i = 0
        while i < doc_len:
            current_word = current_doc_attn[i][0]

            if colorboxed[i]:
                window_num = colorboxed[i]
                current_p_code_latex += '\colorbox{blue!50}{%s ' % current_word

                i += 1
                while i < doc_len:
                    if colorboxed[i] == window_num:
                        current_word = current_doc_attn[i][0]
                        current_p_code_latex += current_word + ' '
                        i += 1
                    else:
                        break

                current_p_code_latex += '} '

            else:
                current_p_code_latex += current_word + ' '
                i += 1

        docs_with_spans.append(current_p_code_latex + '\\newline\n')

    # Concatenate every part to create the full LaTeX output
    full_output = tgt_codes_string + '\\newline\n' + p_codes_string + '\\newline\n'

    for doc in docs_with_spans:
        full_output += doc + '\n'

    tex_file.write(latex_template.replace('YourTextHere', full_output))
    tex_file.close()


def important_spans_colored_raw(data, output, tgt_codes, pred_codes, s, dicts, filter_size,
                                attn_colored_dir, latex_template, df_raw_data, df_spans, hadm_id):
    tex_file = open('%s/%s.tex' % (attn_colored_dir, hadm_id), 'w')
    ind2w, ind2c, desc_dict = dicts['ind2w'], dicts['ind2c'], dicts['desc']

    doc_raw = str(df_raw_data.loc[df_raw_data['HADM_ID'] == hadm_id]['TEXT'].tolist()[0])
    doc_raw_len = len(doc_raw)

    doc_processed = [[ind2w[w], 0] if w in ind2w.keys() else ['UNK', 0] for w in data[0].data.cpu().numpy()]
    doc_processed_len = len(doc_processed)

    # There is a span in spans for each word in doc_proccesed
    # Each span (e.g.: for 'mellitus' in doc_proccesed we have '1312-1320')
    # This span maps to doc_raw in a *character level*, that is, 'mellitus' in
    # doc_proccesed is mapped to, for example, 'Mellitus.' in doc_raw
    # which can be obtained by indexing: doc_proccesed[1312:1320]
    spans = str(df_spans.loc[df_spans['HADM_ID'] == hadm_id]['SPANS'].tolist()[0]).split()
    spans = [tuple(map(int, span.split('-'))) for span in spans]

    palette = np.around(sns.color_palette('pastel', len(pred_codes)), 3)
    colors = {}
    for i, p_code in enumerate(pred_codes):
        colors[p_code] = ','.join(np.around(np.multiply(palette[i], 255)).astype('int').astype('str').tolist())
    doc_raw_attn_windows = []
    for p_code in pred_codes:
        # Get current predicted code's attention vector
        attn = s[0][p_code].data.cpu().numpy()
        # Takes indices of the 10 greatest attention scores
        imps = attn.argsort()[-10:][::-1]

        # Windows are related to doc_proccesed in a *word level*
        # e.g.: ((113, 123), 0.3) goes from word 113 to word
        # 123 in doc_proccesed
        windows = make_windows(imps, filter_size, attn)

        for window_span, window_attn in windows:
            window_start, window_end = window_span
            # window_start + window_size (window_end) can be greater than len(doc)
            window_end = min(window_end, doc_processed_len - 1)

            raw_spans_start = spans[window_start][0]
            raw_spans_end = spans[window_end][1]
            doc_raw_attn_windows.append((raw_spans_start, raw_spans_end, p_code))

    doc_raw_attn_windows = sorted(doc_raw_attn_windows, key=lambda window: window[0])

    # LaTeX codes
    output_codes = ''
    for p_code in pred_codes:
        tp = p_code in tgt_codes
        output_codes += '{icd_code} & \ctext[RGB]{{{color}}}{{{icd_code_desc}}} & {conf} & {type_} \\\\\n'.format(
            icd_code=ind2c[p_code], color=colors[p_code], icd_code_desc=desc_dict[ind2c[p_code]],
            conf=round(output[0][p_code] * 100, 2), type_='TP' if tp else 'FP')

    # LaTeX doc
    output_doc = ''
    written_to = -1
    for window in doc_raw_attn_windows:
        start, end, code = window
        if written_to < start:
            output_doc += doc_raw[written_to + 1:start].replace('\n', '\\newline\n')
            written_to = start

        color = colors[code]
        text = doc_raw[start:end]
        text = text.replace('{', '(')
        text = text.replace('}', ')')
        output_doc += '\ctext[RGB]{{{color}}}{{{text}}}'.format(color=color, text=text).replace('\n', '\\\\\n')
        written_to = end

    if written_to < len(doc_raw):
        output_doc += doc_raw[written_to + 1:].replace('\n', '\\newline\n')

    output_doc = output_doc.replace('>', '$>$')
    output_doc = output_doc.replace('<', '$<$')
    output_doc = output_doc.replace('%', '\%')
    output_doc = output_doc.replace('^', '\^{}')

    output = latex_template
    output = output.replace('YourCodesHere', output_codes)
    output = output.replace('YourDocumentHere', output_doc)
    tex_file.write(output)


def save_attn_plots(data, output, target_data, s, plots_dir, hadm_id, dicts=None):
    output_rd = np.round(output)
    pred_codes = np.where(output_rd[0] == 1)[0]
    tgt_codes = np.where(target_data[0] == 1)[0]
    if dicts is not None:
        if s is not None and len(pred_codes) > 0:
            attn_plots(data, output, tgt_codes, pred_codes, s, dicts, plots_dir, hadm_id)


def attn_plots(data, output, tgt_codes, pred_codes, s, dicts, plots_dir, hadm_id):
    ind2w, ind2c = dicts['ind2w'], dicts['ind2c']

    # Create plots considering predicted codes
    fig, ax = plt.subplots()
    fig.set_dpi(300)

    # Hardcoded for example
    # colors = ['#8080ff', '#ff8080', '#80ff80']
    # for p_code in pred_codes:
    for i in range(len(pred_codes)):
        p_code = pred_codes[i]
        # color = colors[i]

        tp = p_code in tgt_codes
        p_icd_code = ind2c[p_code]
        conf = output[0][p_code]

        # label = '%i [%s]: %s (conf: %f)' % (p_code, "TP" if tp else "FP", p_icd_code, conf)
        label = '{} (conf: {:0.2f})'.format(p_icd_code, conf)

        # Get current predicted code's attention vector
        attn = s[0][p_code].data.cpu().numpy()
        ax.plot(attn, label=label)
        # ax.plot(attn, label=label, color=color)

    # Legends
    ax.set_xlabel('Document k-grams')
    ax.set_ylabel('Attention weight')
    ax.legend()

    # Save
    plt.savefig('%s/%s_pred.png' % (plots_dir, hadm_id))
    plt.close()

    fig, ax = plt.subplots()
    fig.set_dpi(300)
    for p_code in tgt_codes:
        tp = p_code in pred_codes
        p_icd_code = ind2c[p_code]

        if tp:
            conf = output[0][p_code]

        label = '%i: %s' % (p_code, p_icd_code)
        label = '%i [%s]: %s (conf: %s)' % (p_code, "TP" if tp else "FN", p_icd_code, str(conf) if conf else 'NA')

        # Get current predicted code's attention vector
        attn = s[0][p_code].data.cpu().numpy()
        ax.plot(attn, label=label)

    # Legends
    ax.set_xlabel('k-grams')
    ax.set_ylabel('Attention')
    ax.legend()

    # Save
    plt.savefig('%s/%s_tgt.png' % (plots_dir, hadm_id))
    plt.close()


def save_attn_plots_3d(data, output, target_data, s, window_size, plots_dir, hadm_id, dicts=None):
    ind2w, ind2c = dicts['ind2w'], dicts['ind2c']

    kgrams = ['kgram' + str(i) for i in range(s.shape[2])]
    icd_codes = [ind2c[i] for i in range(s.shape[1])]

    doc_words = [ind2w[w] if w in ind2w.keys() else 'UNK' for w in data[0].data.cpu().numpy()]
    doc_len = len(doc_words)
    kgrams = [' '.join(doc_words[i:i + window_size]) for i in range(doc_len)]

    count = 0
    res = []
    for i in range(s.shape[1]):  # 50
        attn = s[0][i].data.cpu().numpy()
        for j in range(s.shape[2] - 1):  # 227
            res.append([kgrams[j], j, icd_codes[i], i, attn[j]])

    df = pd.DataFrame(res, columns=['kgram', 'kgram_num', 'icd_code', 'icd_code_num', 'attn'])

    # X = ICD, Y = ATTN, fixed k-gram
    df_icd_attn = df.loc[df['kgram_num'] == 4]
    fig = plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_icd_attn, x='icd_code_num', y='attn', color='gray')
    ax.set_xticklabels(df_icd_attn['icd_code'].tolist(), rotation='vertical')
    plt.xlabel('ICD-9 code')
    plt.ylabel('Attention')
    plt.tight_layout()
    fig.savefig(plots_dir + '/' + str(hadm_id) + '_2d_icd_attn.png', dpi=300)
    plt.close()

    # X = k-gram, Y = ATTN, fixed ICD
    df_kgrams_attn = df.loc[df['icd_code_num'] == 4]
    fig = plt.figure(figsize=(10, 12))
    ax = sns.barplot(data=df_kgrams_attn, x='kgram_num', y='attn', color='gray')
    kgrams_reduced = [kgram for i, kgram in enumerate(kgrams) if i % 20 == 0]
    plt.xticks(fontsize=8)
    ax.set_xticks(np.arange(0, len(kgrams), 20))
    ax.set_xticklabels(kgrams_reduced, rotation='vertical', Fontsize=12)
    plt.xlabel('k-gram')
    plt.ylabel('Attention')
    plt.tight_layout()
    fig.savefig(plots_dir + '/' + str(hadm_id) + '_2d_kgram_attn.png', dpi=300)
    plt.close()

    # 3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.set_xlim3d(-1, len(kgrams))
    ax.set_xticks(np.arange(0, len(kgrams), 100))

    ax.set_ylim3d(-1, len(icd_codes))
    ax.set_yticks(np.arange(0, len(icd_codes), 5))

    max_ = df['attn'].max()
    df['attn'] = df['attn'] * (1 + max_ - df['attn'])

    ax.plot_trisurf(df['kgram_num'], df['icd_code_num'], df['attn'], cmap=plt.cm.Greys, vmin=0, vmax=0.02)

    kgrams = [kgram for i, kgram in enumerate(kgrams) if i % 100 == 0]
    ax.set_xticklabels(kgrams)

    icd_codes = [icd_code for i, icd_code in enumerate(icd_codes) if i % 5 == 0]
    ax.set_yticklabels(icd_codes)

    ax.set_xlabel('')
    ax.set_ylabel('ICD', fontsize=8)
    ax.set_zlabel('Attention', fontsize=8)

    plt.xticks(rotation=30)
    plt.setp(ax.xaxis.get_majorticklabels(), ha='right')
    ax.tick_params(axis='x', which='major', pad=-7)

    plt.setp(ax.yaxis.get_majorticklabels(), ha='left')
    for tick in ax.get_yticklabels():
        tick.set_verticalalignment("bottom")

    ax.tick_params(axis='y', which='major', pad=-1)
    ax.tick_params(axis='both', which='major', labelsize=5)

    plt.tight_layout()
    plt.gcf().subplots_adjust(left=0.2)
    plt.gcf().subplots_adjust(bottom=0.25)
    plt.savefig(plots_dir + '/' + str(hadm_id) + '_3d.png', dpi=300)
    plt.close()


# def save_samples(data, output, target_data, s, filter_size, tp_file, fp_file, dicts=None):
# Epoch variable was missing in the original version
def save_samples(data, output, target_data, s, filter_size, epoch, tp_file, fp_file, dicts=None):
    """
        save important spans of text from attention
        INPUTS:
            data: input data (text) to the model
            output: model prediction
            target_data: ground truth labels
            s: attention vector from attn model
            filter_size: size of the convolution filter, and length of phrase to extract from source text
            tp_file: opened file to write true positive results
            fp_file: opened file to write false positive results
            dicts: hold info for reporting results in human-readable form
    """
    tgt_codes = np.where(target_data[0] == 1)[0]
    true_str = "Y_true: " + str(tgt_codes)
    output_rd = np.round(output)
    pred_codes = np.where(output_rd[0] == 1)[0]
    pred_str = "Y_pred: " + str(pred_codes)
    if dicts is not None:
        if s is not None and len(pred_codes) > 0:
            important_spans(data, output, tgt_codes, pred_codes, s, dicts, filter_size, true_str, pred_str, tp_file,
                            fps=False)
            important_spans(data, output, tgt_codes, pred_codes, s, dicts, filter_size, true_str, pred_str, fp_file,
                            fps=True)


def important_spans(data, output, tgt_codes, pred_codes, s, dicts, filter_size, true_str, pred_str, spans_file,
                    fps=False):
    """
        looks only at the first instance in the batch
    """
    ind2w, ind2c, desc_dict = dicts['ind2w'], dicts['ind2c'], dicts['desc']
    for p_code in pred_codes:
        # aww yiss, xor... if false-pos mode, save if it's a wrong prediction, otherwise true-pos mode, so save if it's a true prediction
        # Why does it check if >.5? if it's a predicted code, it must be >.5
        if output[0][p_code] > .5 and (fps ^ (p_code in tgt_codes)):
            confidence = output[0][p_code]

            # some info on the prediction
            code = ind2c[p_code]
            conf_str = "confidence of prediction: %f" % confidence
            typ = "false positive" if fps else "true positive"
            prelude = "top three important windows for %s code %s (%s: %s)" % (typ, str(p_code), code, desc_dict[code])
            if spans_file is not None:
                spans_file.write(conf_str + "\n")
                spans_file.write(true_str + "\n")
                spans_file.write(pred_str + "\n")
                spans_file.write(prelude + "\n")

            # find most important windows
            attn = s[0][p_code].data.cpu().numpy()

            # merge overlapping intervals
            # Takes indices of the 10 greatest attention scores
            imps = attn.argsort()[-10:][::-1]
            windows = make_windows(imps, filter_size, attn)
            kgram_strs = []
            i = 0
            while len(kgram_strs) < 3 and i < len(windows):
                (start, end), score = windows[i]
                words = [ind2w[w] if w in ind2w.keys() else 'UNK' for w in data[0][start:end].data.cpu().numpy()]
                kgram_str = " ".join(words) + ", score: " + str(score)
                # make sure the span is unique
                if kgram_str not in kgram_strs:
                    kgram_strs.append(kgram_str)
                i += 1
            for kgram_str in kgram_strs:
                if spans_file is not None:
                    spans_file.write(kgram_str + "\n")
            spans_file.write('\n')


def make_windows(starts, filter_size, attn):
    starts = sorted(starts)
    windows = []
    overlaps_w_next = [starts[i + 1] < starts[i] + filter_size for i in range(len(starts) - 1)]
    overlaps_w_next.append(False)
    i = 0
    get_new_start = True
    while i < len(starts):
        imp = starts[i]
        if get_new_start:
            start = imp
        overlaps = overlaps_w_next[i]
        if not overlaps:
            windows.append((start, imp + filter_size))
        get_new_start = not overlaps
        i += 1
    # return windows sorted by decreasing importance
    # window_scores = {(start,end): attn[start] for (start,end) in windows}
    window_scores = {(start - 1, end - 1): attn[start - 1] for (start, end) in windows}
    window_scores = sorted(window_scores.items(), key=operator.itemgetter(1), reverse=True)
    return window_scores

# def timeline(data, alpha, sections, output, dicts=None):
#    ind2w, ind2c = dicts['ind2w'], dicts['ind2c']
#
#    document = ' '.join([ind2w[w] if w in ind2w.keys() else 'UNK' for w in data[0].data.cpu().numpy()])
#    print('\n\nDOCUMENT\n', document, '\n')
#    tokens_with_timestamps = divide_document(document, sections)
#
#    output_rd = np.round(output)
#    pred_codes = np.where(output_rd[0] == 1)[0]
#    #alpha_per_token = []
#    for pred_code in pred_codes:
#        current_alpha = alpha[0][pred_code][1:]
#    #    alpha_per_token.append(current_v
#        print('\nATTENTION FOR CODE', ind2c[pred_code] ,current_alpha)
#
#    #doc_attn = [[ind2w[w], 0] if w in ind2w.keys() else ['UNK',0] for w in data[0].data.cpu().numpy()]
#    #doc_len = len(doc_attn)
#    #print(len(doc_attn))
#    #print(len(alpha[0][0]))
#    #print(doc_attn)
#    #print(alpha)