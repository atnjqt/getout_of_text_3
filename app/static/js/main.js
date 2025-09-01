document.addEventListener('DOMContentLoaded', function() {
    // Hide collocates filters row on load
    var collocatesFiltersRow = document.getElementById('collocatesFiltersRow');
    if (collocatesFiltersRow) collocatesFiltersRow.style.display = 'none';
    const form = document.getElementById('searchForm');
    const input = document.getElementById('searchInput');
    const dataTable = document.getElementById('dataTable');
    const recordCount = document.getElementById('recordCount');

    // POS tag color map
    const posColorMap = {
        'NN': '#1976d2', 'NNS': '#1976d2', 'NNP': '#1976d2', 'NNPS': '#1976d2', // Nouns: blue
        'VB': '#43a047', 'VBD': '#43a047', 'VBG': '#43a047', 'VBN': '#43a047', 'VBP': '#43a047', 'VBZ': '#43a047', // Verbs: green
        'JJ': '#e65100', 'JJR': '#e65100', 'JJS': '#e65100', // Adjectives: orange
        'RB': '#8e24aa', 'RBR': '#8e24aa', 'RBS': '#8e24aa', // Adverbs: purple
        'IN': '#00838f', // Prepositions: teal
        'CC': '#c62828', // Conjunctions: red
        'DT': '#6d4c41', // Determiners: brown
        'PRP': '#3949ab', 'PRP$': '#3949ab', // Pronouns: indigo
        'CD': '#fbc02d', // Numbers: yellow
        'UH': '#ad1457', // Interjections: pink
        'TO': '#00838f', 'WDT': '#00838f', 'WP': '#00838f', 'WP$': '#00838f', 'WRB': '#00838f', // Wh-words: teal
        'EX': '#455a64', 'FW': '#455a64', 'LS': '#455a64', 'MD': '#455a64', 'PDT': '#455a64', 'POS': '#455a64', 'RP': '#455a64'
    };

    // All POS tags used in backend mapping
    const allPOSTags = [
        'NN','NNS','NNP','NNPS','VB','VBD','VBG','VBN','VBP','VBZ','JJ','JJR','JJS','RB','RBR','RBS','IN','CC','DT','PRP','PRP$','CD','UH','TO','WDT','WP','WP$','WRB','EX','FW','LS','MD','PDT','POS','RP'
    ];

    // Human-readable POS names (should match backend)
    const posNames = {
        'NN': 'noun', 'NNS': 'noun plural', 'NNP': 'proper noun', 'NNPS': 'proper noun plural',
        'VB': 'verb', 'VBD': 'verb past', 'VBG': 'verb gerund', 'VBN': 'verb past participle', 'VBP': 'verb present', 'VBZ': 'verb 3rd person',
        'JJ': 'adjective', 'JJR': 'adj. comparative', 'JJS': 'adj. superlative',
        'RB': 'adverb', 'RBR': 'adv. comparative', 'RBS': 'adv. superlative',
        'IN': 'preposition', 'CC': 'conjunction', 'DT': 'determiner', 'PRP': 'pronoun', 'PRP$': 'possessive pronoun',
        'CD': 'number', 'UH': 'interjection', 'TO': 'to', 'WDT': 'wh-determiner', 'WP': 'wh-pronoun', 'WP$': 'possessive wh-pronoun', 'WRB': 'wh-adverb',
        'EX': 'existential', 'FW': 'foreign word', 'LS': 'list marker', 'MD': 'modal', 'PDT': 'predeterminer', 'POS': 'possessive ending', 'RP': 'particle'
    };

    // Render POS checkboxes for collocates filter (dropdown)
    // Update POS checkboxes rendering to skip first two (reserved for select/deselect all)
    function renderPOSCheckboxes() {
        const menu = document.getElementById('collocatesPOSDropdownMenu');
        // Remove all except first two and divider
        while (menu.children.length > 3) menu.removeChild(menu.lastChild);
        allPOSTags.forEach(tag => {
            const label = posNames[tag] || tag;
            const li = document.createElement('li');
            li.innerHTML = `<label class="dropdown-item"><input type="checkbox" class="form-check-input colloc-pos-chk me-2" value="${tag}" checked> ${tag} <span style="font-size:0.85em;">(${label})</span></label>`;
            menu.appendChild(li);
        });
    }
    renderPOSCheckboxes();

    // Select all/deselect all logic for checkboxes
    document.getElementById('collocatesSelectAllChk').addEventListener('change', function(e) {
        if (e.target.checked) {
            document.querySelectorAll('.colloc-pos-chk').forEach(chk => chk.checked = true);
            document.getElementById('collocatesUnselectAllChk').checked = false;
        }
    });
    document.getElementById('collocatesUnselectAllChk').addEventListener('change', function(e) {
        if (e.target.checked) {
            document.querySelectorAll('.colloc-pos-chk').forEach(chk => chk.checked = false);
            document.getElementById('collocatesSelectAllChk').checked = false;
        }
    });

    function renderTable(data, query = '') {
        if (!data.length) {
            dataTable.innerHTML = '<div class="alert alert-warning">No records found.</div>';
            recordCount.textContent = '';
            return;
        }
        let html = '<table class="table table-striped table-bordered"><thead><tr>';
        const cols = ['Docket No.', 'Granted', 'Argued', 'Decided', 'url', 'oyez', 'snippet'];
        for (const col of cols) html += `<th>${col === 'snippet' ? 'Context Snippet' : col === 'oyez' ? 'Oyez' : col}</th>`;
        html += '</tr></thead><tbody>';
        for (const row of data) {
            html += '<tr>';
            for (const col of cols) {
                if (col === 'url') {
                    html += `<td><a href="${row[col]}" target="_blank" rel="noopener">View PDF</a></td>`;
                } else if (col === 'oyez') {
                    // Extract year from Granted (assume format contains 4-digit year)
                    let year = '';
                    if (row['Granted']) {
                        const match = row['Granted'].match(/(\d{4})/);
                        if (match) year = match[1];
                    }
                    let docket = row['Docket No.'] || '';
                    let oyezUrl = '';
                    if (year && docket) {
                        oyezUrl = `/oyez?year=${encodeURIComponent(year)}&docket=${encodeURIComponent(docket)}`;
                        html += `<td><a href="${oyezUrl}" rel="noopener">Oyez</a></td>`;
                    } else {
                        html += '<td></td>';
                    }
                } else if (col === 'snippet') {
                    let snippet = row[col] || '';
                    if (query) {
                        // Highlight all matches of the query (case-insensitive)
                        const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\\]\\]/g, '\\$&')})`, 'gi');
                        snippet = snippet.replace(regex, '<span class="highlighted-term">$1</span>');
                    }
                    html += `<td><span class="snippet">${snippet}</span></td>`;
                } else {
                    html += `<td>${row[col] || ''}</td>`;
                }
            }
            html += '</tr>';
        }
        html += '</tbody></table>';
        dataTable.innerHTML = html;
        recordCount.textContent = `Showing ${data.length} records.`;
    }

    function fetchData(query) {
        if (query) {
            fetch(`/search?query=${encodeURIComponent(query)}`)
                .then(resp => resp.json())
                .then(data => {
                    if (data.message || data.error) {
                        renderTable([], query);
                    } else {
                        renderTable(data, query);
                    }
                });
        } else {
            fetch('/data.json')
                .then(resp => resp.json())
                .then(data => {
                    data.forEach(row => {
                        if (row.text && row.text.length > 100) {
                            const mid = Math.floor(row.text.length / 2);
                            row.snippet = '...' + row.text.substring(Math.max(0, mid - 50), Math.min(row.text.length, mid + 50)).replace(/\n/g, ' ') + '...';
                        } else {
                            row.snippet = row.text || '';
                        }
                    });
                    renderTable(data, '');
                });
        }
    }

    // KWIC rendering
    function renderKwicTable(data, query) {
        const isDark = document.body.classList.contains('dark-mode');
        const kwicClass = isDark ? 'alert-warning' : 'alert-info';
        if (!data.length) {
            dataTable.innerHTML = `<div class="alert ${kwicClass}">No KWIC results found.</div>`;
            recordCount.textContent = '';
            return;
        }
        let html = `<table class="table table-striped table-bordered"><thead><tr><th style="width:40%">Left</th><th style="width:20%; white-space:nowrap; text-align:center;">Keyword</th><th style="width:40%">Right</th></tr></thead><tbody>`;
        for (const row of data) {
            let keywordHtml = row.keyword;
            if (query) {
                // Highlight all matches of the query in the keyword column (case-insensitive)
                const safeQuery = query.replace(/[.*+?^${}()|[\\]\/]/g, '\\$&');
                const regex = new RegExp('(' + safeQuery + ')', 'gi');
                keywordHtml = keywordHtml.replace(regex, '<span class="highlighted-term">$1</span>');
            }
            html += `<tr><td class="text-end kwic-left">${row.left}</td><td class="kwic-keyword text-center">${keywordHtml}</td><td class="kwic-right">${row.right}</td></tr>`;
        }
        html += '</tbody></table>';
        dataTable.innerHTML = html;
        recordCount.textContent = `KWIC: ${data.length} hits.`;
    }

    // Collocates rendering (no POS by default, async POS load)
    // Show/hide the POS load button in the fixed box above the table
    function showCollocPOSLoadBtn(show, data) {
        const box = document.getElementById('collocatesPOSLoadBox');
        if (show) {
            box.style.display = '';
            let btn = document.getElementById('loadCollocPOSBtn');
            if (!btn) {
                btn = document.createElement('button');
                btn.id = 'loadCollocPOSBtn';
                btn.className = 'btn btn-outline-info';
                btn.innerHTML = '<i class="fas fa-magic me-1"></i>Load Part of Speech Tags';
                box.appendChild(btn);
            }
            btn.onclick = function() { loadCollocatesPOS(data); };
        } else {
            box.style.display = 'none';
            const btn = document.getElementById('loadCollocPOSBtn');
            if (btn) btn.remove();
        }
    }

    function renderCollocatesTable(data, posData = null) {
        // Get selected POS tags
        const selectedPOS = Array.from(document.querySelectorAll('.colloc-pos-chk:checked')).map(chk => chk.value);
        let filtered = data;
        let posMap = {};
        if (posData) {
            posData.forEach(row => { posMap[row.word] = row.pos; });
            filtered = data.filter(row => {
                const tag = (posMap[row.word] || '').split(' ')[0];
                return selectedPOS.includes(tag);
            });
        }
        if (!filtered.length) {
            dataTable.innerHTML = '<div class="alert alert-warning">No collocates found for selected POS.</div>';
            recordCount.textContent = '';
            return;
        }
        let html = '<table class="table table-striped table-bordered"><thead><tr><th>Word</th>';
        if (posData) html += '<th>POS</th>';
        html += '<th>Count</th></tr></thead><tbody>';
        for (const row of filtered) {
            html += `<tr><td class="colloc-word">${row.word}</td>`;
            if (posData) {
                const tag = (posMap[row.word] || '').split(' ')[0];
                let posClass = '';
                if (["NN","NNS","NNP","NNPS"].includes(tag)) posClass = 'pos-info';
                else if (["VB","VBD","VBG","VBN","VBP","VBZ"].includes(tag)) posClass = 'pos-danger';
                else if (["RB","RBR","RBS"].includes(tag)) posClass = 'pos-warning';
                else if (["JJ","JJR","JJS"].includes(tag)) posClass = 'pos-success';
                else posClass = 'pos-secondary';
                html += `<td><span class="pos-badge ${posClass}">${posMap[row.word] || ''}</span></td>`;
            }
            html += `<td class="colloc-count">${row.count}</td></tr>`;
        }
        html += '</tbody></table>';
        dataTable.innerHTML = html;
        recordCount.textContent = `Top ${filtered.length} collocates.`;
        // Show POS load button if not loaded
        if (!posData) {
            showCollocPOSLoadBtn(true, data);
        } else {
            showCollocPOSLoadBtn(false);
        }
    }

    async function loadCollocatesPOS(data) {
        const words = data.map(row => row.word);
        const resp = await fetch('/collocates-pos', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ words })
        });
        const posData = await resp.json();
        renderCollocatesTable(data, posData);
    }

    // Custom POS badge styles (Bootstrap-like)
    const style = document.createElement('style');
    style.innerHTML = `
    .pos-badge { display: inline-block; padding: 0.25em 0.7em; border-radius: 0.7em; font-size: 0.95em; font-weight: 600; border: 1.5px solid; background: transparent; }
    .pos-info { color: #0d6efd; border-color: #0d6efd; background: rgba(13,110,253,0.08); }
    .pos-danger { color: #dc3545; border-color: #dc3545; background: rgba(220,53,69,0.08); }
    .pos-warning { color: #ffc107; border-color: #ffc107; background: rgba(255,193,7,0.10); }
    .pos-success { color: #198754; border-color: #198754; background: rgba(25,135,84,0.10); }
    .pos-secondary { color: #6c757d; border-color: #6c757d; background: rgba(108,117,125,0.10); }
    `;
    document.head.appendChild(style);

    // Re-render table on POS filter change
    document.getElementById('collocatesPOSDropdownMenu').addEventListener('change', function() {
        // Re-run collocates button click to update table
        document.getElementById('collocatesBtn').click();
    });

    // Button event listeners
    document.getElementById('kwicBtn').addEventListener('click', function() {
        // Show KWIC filters, hide collocates filters
        document.getElementById('kwicCharWindowBox').style.display = '';
        document.getElementById('collocatesPOSFilterBox').style.display = 'none';
        // Hide collocates filters row if present
        var collocatesFiltersRow = document.getElementById('collocatesFiltersRow');
        if (collocatesFiltersRow) collocatesFiltersRow.style.display = 'none';
        const query = input.value.trim();
        if (!query) return;
        const leftChars = document.getElementById('kwicLeftChars').value || 50;
        const rightChars = document.getElementById('kwicRightChars').value || 50;
        dataTable.innerHTML = '<div class="text-center text-muted">Loading KWIC...</div>';
        fetch(`/kwic?query=${encodeURIComponent(query)}&left_chars=${leftChars}&right_chars=${rightChars}`)
            .then(resp => resp.json())
            .then(data => renderKwicTable(data, query));
    });
    document.getElementById('collocatesBtn').addEventListener('click', function() {
        // Show collocates filters, hide KWIC filters
        document.getElementById('collocatesPOSFilterBox').style.display = '';
        document.getElementById('kwicCharWindowBox').style.display = 'none';
        // Show collocates filters row if present
        var collocatesFiltersRow = document.getElementById('collocatesFiltersRow');
        if (collocatesFiltersRow) collocatesFiltersRow.style.display = 'flex';
        // Hide KWIC-specific table and record count if present
        if (typeof renderKwicTable === 'function') {
            dataTable.innerHTML = '';
            recordCount.textContent = '';
        }
        const query = input.value.trim();
        if (!query) return;
        const topN = document.getElementById('collocatesTopN') ? document.getElementById('collocatesTopN').value : 30;
        const windowVal = document.getElementById('collocatesWindow') ? document.getElementById('collocatesWindow').value : 5;
        dataTable.innerHTML = '<div class="text-center text-muted">Loading collocates...</div>';
        fetch(`/collocates?query=${encodeURIComponent(query)}&window=${windowVal}&top_n=${topN}`)
            .then(resp => resp.json())
            .then(data => renderCollocatesTable(data));
    });

    form.addEventListener('submit', function(e) {
        e.preventDefault();
        fetchData(input.value);
    });

    // Initial load
    fetchData('');
});
