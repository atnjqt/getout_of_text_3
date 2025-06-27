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

    form.addEventListener('submit', function(e) {
        e.preventDefault();
        fetchData(input.value);
    });

    // Initial load
    fetchData('');
});
