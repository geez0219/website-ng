import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { HttpClient } from '@angular/common/http';

import { GlobalService } from '../global.service';

@Component({
  selector: 'app-search',
  templateUrl: './search.component.html',
  styleUrls: ['./search.component.css']
})
export class SearchComponent implements OnInit{
  searchText: string;

  constructor(private http: HttpClient,
              private router: Router,
              private globalService: GlobalService) {}

  ngOnInit() {}

  search() {
    this.router.navigate(['/searchresult'], { queryParams: { query: this.searchText, version: this.globalService.getSelectedVersion() } });
  }
}
