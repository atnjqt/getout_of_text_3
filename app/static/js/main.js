document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('searchForm');
    const input = document.getElementById('searchInput');
    const dataTable = document.getElementById('dataTable');
    const recordCount = document.getElementById('recordCount');

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
        if (!data.length) {
            dataTable.innerHTML = '<div class="alert alert-warning">No KWIC results found.</div>';
            recordCount.textContent = '';
            return;
        }
        let html = '<table class="table table-striped table-bordered"><thead><tr><th>Left</th><th>Keyword</th><th>Right</th></tr></thead><tbody>';
        for (const row of data) {
            html += `<tr><td class="text-end kwic-left">${row.left}</td><td class="kwic-keyword">${row.keyword}</td><td class="kwic-right">${row.right}</td></tr>`;
        }
        html += '</tbody></table>';
        dataTable.innerHTML = html;
        recordCount.textContent = `KWIC: ${data.length} hits.`;
    }

    // Collocates rendering
    function renderCollocatesTable(data) {
        if (!data.length) {
            dataTable.innerHTML = '<div class="alert alert-warning">No collocates found.</div>';
            recordCount.textContent = '';
            return;
        }
        let html = '<table class="table table-striped table-bordered"><thead><tr><th>Word</th><th>Count</th></tr></thead><tbody>';
        for (const row of data) {
            html += `<tr><td class="colloc-word">${row.word}</td><td class="colloc-count">${row.count}</td></tr>`;
        }
        html += '</tbody></table>';
        dataTable.innerHTML = html;
        recordCount.textContent = `Top ${data.length} collocates.`;
    }

    // Button event listeners
    document.getElementById('kwicBtn').addEventListener('click', function() {
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
        const query = input.value.trim();
        if (!query) return;
        dataTable.innerHTML = '<div class="text-center text-muted">Loading collocates...</div>';
        fetch(`/collocates?query=${encodeURIComponent(query)}&window=5`)
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
