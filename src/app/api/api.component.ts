import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http'; 

@Component({
  selector: 'app-api',
  templateUrl: './api.component.html',
  styleUrls: ['./api.component.css']
})
export class ApiComponent implements OnInit {

  apiList: string[];
  currentSelection: string;
  currentAPI: string;

  constructor(private http: HttpClient) { }

  ngOnInit() {
    this.apiList = ['network', 'pipeline'];
    this.currentSelection = 'assets/api/network.md';
    
    this.getAPIText();

    console.log(this.currentAPI);
  }

  updateCurrentAPI(api: string) {
    this.currentSelection = 'assets/api/' + api + '.md';

    this.getAPIText();
  }

  getAPIText() {
    this.http.get(this.currentSelection, {responseType: 'text'}).subscribe(data => {
      this.currentAPI = data;
    });
  }

}
