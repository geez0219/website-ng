import { Component, OnInit }  from '@angular/core';
import { Router } from '@angular/router';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-search',
  templateUrl: './search.component.html',
  styleUrls: ['./search.component.css']
})
export class SearchComponent implements OnInit{
  searchText: string;

  constructor(private http: HttpClient,
              private router: Router) {}

  ngOnInit() {}

  search() {
    console.log("searching for " + this.searchText);
    this.router.navigate(['/searchresult'], { queryParams: { query: this.searchText } });
  }
}
